# Improving-Closed-loop-Matters(Super Resolution)

[Presentation](https://docs.google.com/presentation/d/1AwdgyuRATms0tHZfOAdPIokXjWx52tRM/edit?usp=sharing&ouid=113730967271079117176&rtpof=true&sd=true)

## Goal
By adding a small idea, tried to improve **Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution(CVPR 2020)** baseline code

## Overview

![제목 없음](https://user-images.githubusercontent.com/65278309/165081476-5481d638-f9df-4d02-89c8-37c01626a3ee.png)

Inspired by SamsungSDS 's solution to [Frequency based detecting Deepfake](https://www.aaai.org/AAAI22Papers/AAAI-1171.JeongY.pdf),<br/><br/>
Images in Frequency domain seems to contain extra infomations that are not in rgb images.<br/><br/>
Thus I used **images in frequency domain** and rgb images at the sametime to train model for Super Resolution task<br/><br/>
I simply calcuated new Floss(frequency loss), which is  loss between GT images in frequency doamin and created images in frequency domain<br/><br/>
As result there was a slight improvement in PSNR score.

## Results
![image](https://user-images.githubusercontent.com/65278309/165083555-9ed58b6e-544b-4538-9ba1-1be297b072c3.png)
Tried simple tests altering weights of Floss, and there were slight improvement, but it was hard to tell with bare eyes

## Ground Truth
![image](https://user-images.githubusercontent.com/65278309/165083909-8c0effa3-46c0-4934-b42e-034ab0428a88.png)
![image](https://user-images.githubusercontent.com/65278309/165083920-8f866c8b-5b09-4871-b982-584576c95787.png)
## Baseline
![image](https://user-images.githubusercontent.com/65278309/165083932-c6952714-20d8-4957-af74-3e309009fb5b.png)
![image](https://user-images.githubusercontent.com/65278309/165083937-5c483f62-5f79-4fb8-b278-9ef84a993bb6.png)
## With New Floss
![image](https://user-images.githubusercontent.com/65278309/165083948-338571f9-3004-4214-935e-48d4ab91089b.png)
![image](https://user-images.githubusercontent.com/65278309/165083958-4c8731f4-9400-4057-a67f-e8ff0f24a644.png)


## Reference
Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution [[arXiv]](https://arxiv.org/pdf/2003.07018.pdf) [[github]](https://github.com/guoyongcs/DRN)



