# Skin Detective
---

References:


https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


https://arxiv.org/pdf/1405.0312.pdf


--- 
## Get Started

### Container

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

--- 
## How to use

**Folder structure** 


**Setup**

1. Install requirement packages
```
pip install -r requirements.txt
```

2. Prepare dataset
Object detection dataset should be organized in COCO format
```
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


3. Train object detection
```
python acne_detection.py
```
Look at help for more detail parameters
Model will be trained and stored in **./models** folder

4. Train Grade Classifier

Run acne_circle_final.ipynb
This should output lightgbm model for grade classifer

---

## Demo

Run file present2.ipynb to present the result and evaluation