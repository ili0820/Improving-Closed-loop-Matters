import torch
import utility
from decimal import Decimal
from tqdm import tqdm
import numpy as np
from numpy.fft import fft2
import cv2
class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            sr = self.model(lr[0])
            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # compute primary loss\
            loss_primary = self.loss(sr[-1], hr)

            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

            # compute dual loss
            tmp = hr.cpu()
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])


            # compute hr bright channel loss
            temp1 = np.zeros((1, tmp.shape[2], tmp.shape[3]))
            for _ in range(tmp.shape[0]):
                swap = np.transpose(tmp[_], (1, 2, 0))
                b, g, r = cv2.split(np.array(swap))
                for i in range(temp1.shape[1]):
                    for j in range(temp1.shape[2]):
                        temp1[0][i][j] = max(b[i][j], g[i][j], r[i][j])


            # compute sr bright channel loss
            tmp = sr[-1].cpu()
            temp2 = np.zeros((1, tmp.shape[2], tmp.shape[3]))
            for _ in range(tmp.shape[0]):
                swap = np.transpose(tmp[_].detach().numpy(), (1, 2, 0))
                b, g, r = cv2.split(np.array(swap))
                for i in range(temp2.shape[1]):
                    for j in range(temp2.shape[2]):
                        temp2[0][i][j] = max(b[i][j], g[i][j], r[i][j])

            brchannelloss=self.loss(torch.Tensor(temp1),torch.Tensor(temp2))

            #loss triple
            fsr=[]
            flr=[]
            fsr2lr=[]



            #HR to Frequency
            temp = hr.cpu()

            fhr= torch.from_numpy(abs(fft2(temp)/len(temp) )).to("cuda")
            #SR to Frequency
            for _ in range(len(sr)):
                temp=sr[_].cpu()
                fsr.append(torch.from_numpy(abs(fft2(temp.detach())/len(temp))).to("cuda") )
            #SR2LR to Frequency
            for _ in range(len(sr2lr)):
                temp=sr2lr[_].cpu()
                fsr2lr.append(torch.from_numpy(abs(fft2(temp.detach())/len(temp))).to("cuda") )
            #LR to Frequency
            for _ in range(len(lr)):
                temp=lr[_].cpu()
                flr.append(torch.from_numpy(abs(fft2(temp.detach())/len(temp))).to("cuda") )

            floss_primary = self.loss(fsr[-1], fhr)
            for i in range(1, len(fsr)):
                floss_primary += self.loss(fsr[i - 1 - len(fsr)], flr[i - len(fsr)])

            floss_dual = self.loss(fsr2lr[0], flr[0])
            for i in range(1, len(self.scale)):
                floss_dual += self.loss(fsr2lr[i], flr[i])
            # print(type(fhr),type(hr))
            # print(fhr.shape,hr.shape)

            # compute total loss



            loss = (loss_primary + self.opt.dual_weight *(loss_dual))+(floss_primary+ self.opt.dual_weight *(floss_dual)) + 0.1*brchannelloss
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)

                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
