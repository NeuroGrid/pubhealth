# Task:

Build an ML system to verify the veracity of claims. Please clearly outline steps and demonstrate the performance of the final model.

While we strongly encourage creativity in this challenge, you might find some of the pre-trained transformers available in Huggingface helpful.

Huggingface transformers: https://huggingface.co/models

# Dataset:

PUBHEALTH is a comprehensive dataset for explainable automated fact-checking of public health claims. Each instance in the PUBHEALTH dataset has an associated veracity label (true, false, unproven, mixture). Furthermore each instance in the dataset has an explanation text field. The explanation is a justification for which the claim has been assigned a particular veracity label.

Dataset link: https://huggingface.co/datasets/health_fact

## Executive Summary

Best performance was from a longformer trained on custom feature that combined the text in the existing feaures subjects, claim and explanation.

## Approach

The approach I took started with an [Exploratory Data Analysis](pubhealth_eda.ipynb) which indicated that the dataset had an undocumented "invalid" label that we needed to remove, as well as indicating that the classes were somewhat imbalanced.  I saw that the words per explanation were an appropriate amount for the 512 limit on most transformers, and that otherwise the various features in the dataset did not obviously distinguish the veracity classes.  There was one exception in that the subject of the claim was quite skewed in some cases.  For example all the claims in the "Health" and "General news" subjects were true, while the majority of the claims in the "Coronavirus" are false.

Next I tried the [distilbert transformer on just the explanation text](pubhealth_distilbert.ipynb), and created a visualization of the different classes based on the distilbert embeddings.  This showed that the classes did seem somewhat distinguishable, and I proceeded to train a logistic regression model on the distilbert embeddings, which gave an accuracy of 0.684.  This compared favourably to a dummy classifier which got only 0.513. Tuning the distilbert model took us to an accuracy of ~0.73. Further analysis showed that the model was having trouble dealing with the unproven category.

A detailed looked at some of the claims with the highest losses indicated that some short explanations didn't make much sense in the absence of information about the claim.  Also it seemed that some claims were completely unrelated to the explanations.

This suggested several possible approaches to improve the accuracy of the classifier:

1. rebalancing the classes
2. including information about the subject
3. including information about the claim
4. using a transformer that could handle more tokens
5. hyperparameter searches to optimise the training process

Rebalancing the classes using class weights did not improve the performance particularly, and hyperparameter searches around learning_rate, batch_size, weight_decay, warmup_step didn't fair much better.  That said, particular combinations did push the accuracy up to ~0.75.

Combining some of the possible approaches I selected on creating a new feature comprised of subject, claim and explanation that looked like this:

```python
f"SUBJECTS: {example['subjects']} - CLAIM: {example['claim']} - EXPLANATION: {example['explanation']}"
```

The idea being to allow the transformer to distinguish the different components of the feature.  Furthermore in order to cope with the increased text length I selected the longformer, which can handle up to 4096 tokens and so we could now comforytably include the subjects, claim and even the longest explanation.

The [longformer approach](pubhealth_longformer.ipynb) produced better results in training, up to ~0.79, although it did take signiificantly longer to train (~2 hours).  This model gives a final accuracy of 0.742 on the hold out test set, indicating reasonable generalization performance to unseen data.


## Future Work

With more time there'd be a few other avenues that I'd like to explore.

* further cleaning of the data to remove claims that have no relation to their texts or explanation
* include main_text in longformer
* alternate resampling strategies
* "hyperparameter" search over resampling strategies and transformer types

