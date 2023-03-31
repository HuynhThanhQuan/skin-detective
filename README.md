# Skin Detective AI Engine

This is the AI core engine behind the project Skin-Detective AI which is published on Diagnostics ([Scopus Impact Factor 3.9](https://www.mdpi.com/journal/diagnostics/imprint))
- https://www.mdpi.com/2075-4418/12/8/1879 
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9406819/ 

![z3756639763317_f42efb611bb2356a668a7cef762a4118](https://user-images.githubusercontent.com/22089209/229078371-3fccfbde-407e-4e47-a577-95dc4e07eefb.jpg)

**News**
- https://tuoitre.vn/khoi-nghiep-voi-khat-vong-khong-de-nguoi-dan-hoi-chut-la-di-vien-20230321231316476.htm?gidzl=pJqKRTrv97FA8H9eqsSdSC4W5cpEOY8aZd4NEfzb97ME84K-dMOc8umb5plEQYKZt2KIQp0XTWSjtNCdT0
- https://cafef.vn/uoc-mo-nguoi-o-nong-thon-van-duoc-kham-bac-si-gioi-dang-sau-ung-dung-ai-ho-tro-kham-benh-danh-rieng-cho-nguoi-viet-20221120161048606.chn
- https://vnexpress.net/5-du-an-duoc-dau-tu-ai-tech-matching-4514085.html?gidzl=5dFa8v9OWouHR-8cd4Q0BHfcw2A1IujvLJVfTD1CZIXLPhjmqa-FVbTZwNM1Gun-1sFi9J2r6L0-abA0AG
- https://vnexpress.net/skin-detective-ung-dung-tich-hop-tri-tue-nhan-tao-phat-hien-cac-benh-ve-da-va-ket-noi-bac-si-da-lieu-4498851.html
- https://drive.google.com/file/d/1_tZqrh5ARUuLCWThNOvOH1k30iosA1LL/view


**Reference**
- https://arxiv.org/pdf/1405.0312.pdf

--- 
## Get Started

### 1. Container

Reproduce the exact environment
```
docker pull hthquan28/skin-detective

```

Run Jupyterlab in container, with expose port 8080 and using NVIDIA Driver
```
docker run -it --gpus all -p 8080:8080 --name skin_container hthquan28/skin-detective
```

Run on detached mode
```
docker run -d --gpus all -p 8080:8080 --name skin_container hthquan28/skin-detective
```

Using terminal in container
```
docker exec -it skin_container bash
```

Build container
```
sh build_local.sh
```

*Note*: 
Due to different architectures of your Graphic Card, you might not able to run it, for more information please refer this article. https://github.com/NVIDIA/nvidia-docker

## 2. Setup

### 2.1 Install requirement packages
```
pip install -r requirements.txt
```

### 2.2 Prepare dataset
Object detection dataset should be organized in COCO format
```bash
root
| - bbox			<contain COCO format files info>
   	| - img1		<coco info of img1>
   	| - img2		<coco info of img2>
| - image			<contain images>
	| - img1.jpg		<img1>
   	| - img2.jpg		<img2>
| - models			<store model files>
| - mAP				<eval package>
```	

### 2.3 Train object detection
```
python acne_detection.py
```
Look at help for more detail parameters
Model will be trained and stored in **./models** folder

### 2.4 Train Grade Classifier
Run acne_circle_final.ipynb
This should output lightgbm model for grade classifer

## 3. Demo

Run file present2.ipynb to present the result and evaluation
