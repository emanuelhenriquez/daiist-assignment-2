# Assignment 2 Report

- **Name**: Emanuel Cohen-Henriquez
- **Student ID**: 18761
- **Email**: ecohenhenriq.ieu2022@student.ie.edu
- **Group**:  BBADBA 5B

## Business problem, and the cost of being wrong

The model created will help a clinic that uses x-rays to look for pneumonia cases (bacterial or viral). Uploading an x-ray image, the model will classify it in three different levels: pass (normal), review (a radiologist reads it) or fail (pneumonia, start treatment). There are three costs to consider: 

- Missed pneumonia, a false negative: €1,500. The patient is sent home, comes back sicker, and may be admitted.
- False alarm, a false positive: €100, for unnecessary antibiotics and a follow-up visit.
- Radiologist review: €10, about 3 minutes of a specialist.

Meaning a missed pneumonia is 15× worse than a false alarm. Because a miss cost is way higher, the model should flag generously (a low threshold), and unclear cases go to a radiologist. Doctors act on the result, and radiologists read the cases sent to review.

To count True positives, True negatives, False positives, and False negatives, I decided that the positive class will be having pneumonia, and the negative class not having it.


## Dataset

The dataset is Roboflow inatel/pneumonia-classification-imrcv, version 1 (https://universe.roboflow.com/inatel/pneumonia-classification-imrcv), CC BY 4.0. One image = one chest X-ray, resized to 640×640. 3 classes: Normal, Pneumonia_bacteria, Pneumonia_virus.

This dataset serves the problem because it contains real chest X-rays, which is exactly the image a clinic's doctor looks at when pneumonia is suspected, and its labels match the decision: Normal maps to "pass", and both pneumonia types map to "fail". It also separates bacterial from viral pneumonia, which matters for treatment.

## What the dataset's composition implies

The dataset was composed of 5,190 images, split into: Normal 1,340, bacterial 2,512, viral 1,338. 74% of the cases had pneumonia, but I assumed 25% in real use, as the number was way too high so I could make model learn how to identify it, and 25% is closer to the real amount of cases. That’s why every cost is re-weighted to the 25% base rate. 

The published version has no split (everything is in train). My code splits 70/15/15 by patient: 2,846 patients, 1,074 of whom have several X-rays. That keeps the same patient out of both train and test. Result: train 3,675, val 756, test 759 images. The test split has 74% pneumonia, not the 25% of real use, which is why I re-weight the costs.

The biggest limitation is that all images come from one source, so the model may not hold for other hospitals’ machines. Also the test is fairly small, as there were 194 normal and 203 viral, so one image changes a class’s recall by about 0.5 percentage points. So differences of one or two points between results are not meaningful.

In terms of object sizes, since this is classification, the label covers the whole image; all images are 640×640.

## Finetuning and baselines
I finetuned yolo11n-cls (image classification) at 224 px for 15 epochs, batch 32, on my Apple M2 laptop (MPS). I chose classification rather than detection because the decision is about the whole X-ray (does this patient have pneumonia?), not about drawing boxes around objects, and the dataset gives one label per image.

The test results showed:

| Model | Accuracy (3 classes) | Pneumonia sensitivity | Specificity |
|---|---|---|---|
| Finetuned yolo11n-cls | 0.793 | 0.947 | 0.985 |
| Trivial: always "bacterial" | 0.477 | 1.000 | 0.000 |
| Zero-shot pretrained yolo11n-cls (ImageNet) | 0.586 | 0.903 | 0.588 |

For classification, the pretrained model is trained on ImageNet, not COCO, so it cannot output my classes at all. On the test X-rays its most common guess was "radiator" (358 images). To still use it as a zero-shot baseline, without training any weights, I mapped each ImageNet guess to the label it most often had in the training split. It scores better than the trivial baseline but still calls about 4 in 10 healthy patients sick. Finetuning made the biggest difference on specificity (0.588 to 0.985) and on accuracy (0.586 to 0.793). The gap was larger than I expected, because the pretrained model has never seen an X-ray.

## Per-class evaluation and failure modes

| Class | Test images | Precision | Recall |
|---|---|---|---|
| Normal | 194 | 0.864 | 0.985 |
| Pneumonia_bacteria | 362 | 0.776 | 0.873 |
| Pneumonia_virus | 203 | 0.725 | 0.468 |

Confusion matrix of the finetuned model on the test split (rows = actual, columns = predicted):

| Actual \ Predicted | Normal | Bacterial | Viral |
|---|---|---|---|
| Normal | 191 | 1 | 2 |
| Bacterial | 12 | 316 | 34 |
| Viral | 18 | 90 | 95 |

The biggest failure was that 90 of the 203 viral cases were called bacterial, so viral recall is only 47%.

As the model count both cases as pneumonia, the patient is still flagged. But treatment differs, since antibiotics are for bacterial pneumonia, so this matters for choosing the treatment, not for the triage decision.

The dangerous error is 30 pneumonia cases called Normal at the default decision (18 viral, 12 bacterial).


## Cost matrix and operating point

The costs are my assumptions. A missed pneumonia (€1,500) means the patient returns sicker and may be admitted; a false alarm (€100) means unnecessary antibiotics and a follow-up; a review (€10) is 3 minutes of a radiologist. Correct decisions cost €0. I assume 25% of X-rays show pneumonia in real use (74% in the dataset), so all rates are re-weighted to 25%.

I swept every threshold from 0 to 1 on validation (plot in the dashboard's "Cost curve" tab). For the model alone the cheapest is 0.04: €9,337 per 1,000 X-rays on test, against €19,576 at 0.5. It is so low because a miss costs 15× a false alarm. The review thresholds (0.03 and 0.84) are robust: they stay the same for miss costs from €300 to €3,000 and base rates from 10% to 40%. The assumption that matters most is that the radiologist is always right.

## What a person still has to check

The model decides the clear cases: below 0.03 is a pass and from 0.84 is a fail. The X-rays in between go to a radiologist, who makes the decision. 10.1% of X-rays go to review: 15 a day out of 150, × 3 minutes = 0.76 hours/day. That’s 0.13 of a radiologist, which is realistic.

On the test split, only 3 of 565 pneumonia cases are missed (all viral), and 0 false alarms.

If we check the alternatives:
- Do nothing (everyone passes) = €375,000
- Radiologist reads everything = €10,000 (1.25 radiologists)
- Model alone = €9,337
- Model + review = €3,003

Model + review is about 3× cheaper than the model alone or a radiologist reading everything, and it only needs 0.13 of a radiologist.

## Limitations & next steps

-The most important one is that the radiologist is assumed perfect. This will probobly not be the case in real life, as everybody makes mistakes.

-As I only used one data source (and not a large one) the model could be biased. Even more so on the test set, as it is a small one.

-It does not distincts well between Viral and bacterial cases (finds 47% of viral cases), which could create confusion or affect accuracy on treatment on real life.

-The Base rate was chosen by me, and its not measured.

The best fix would be to use real hospital data, measure radiologist accuracy, add external test sets, train longer or a bigger model (yolo11s-cls) for viral cases.

## Generative AI use disclosure

Claude found the dataset options. I chose the problem and confirmed the costs and volumes. Claude did the heavy coding side while I ran the training on my laptop and did all the report with Claude checking it.
