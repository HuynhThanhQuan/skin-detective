# Skin Detective

References:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://arxiv.org/pdf/1405.0312.pdf


--- 
## Outline

- Setup environment
- How to use
  - Object Detection
  - Machine Learning
- Demo

--- 

## Environment

```
python>=3.7
```

---
## How to use

**Folder structure** 


- Object detection

  - acne_detection.py
  - coco_eval.py
  - coco_utils.py
  - engine.py
  - transform.py
  - utils.py

- Machine learning
  - acne_circle_final.ipynb

- Present/Demo
  - present3.ipynb

- mAP


**Setup**

1. Install requirement packages
```
pip install -r requirements.txt
```

2. Prepare dataset
Object detection dataset should be organized in COCO format
```
root
| - bbox	<contain COCO format files info>
| - image	<contain images>
| - models	<store model files>
| - mAP		<eval package>
```	


3. Train object detection
```
python acne_detection.py
```
Look at help for more detail parameters
Model will be trained and stored in **./models** folder

4. Train Grade Classifier

Run acne_circle_final.ipynb

---

## Demo

Run file present2.ipynb`