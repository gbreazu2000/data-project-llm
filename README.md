# LLM Project

## Project Task
The goal of this project was to take movie reviews from the imdb website and create a text sentiment analysis model that can determine whether or not a review was negative or positive.

## Dataset
The dataset was imported from the imdb dataset in huggingface. I manually preprocessed the review text data to learn more about the process. I initially removed stopwords and punctuation. Then, I tokenized the reviews by separating each review into a list of individual words and continued by using the functions CountVectorizer() and word2vec for capturing information around the word meanings.

## Pre-trained Model
From hugging face, I imported a pipeline containing a pre-trained model into my Google Collab for text-classification. This model can automatically preprocess the data, so all I had to do was evaluate the imdb reviews using the pre-trained model.

## Hyperparameters
For the hyperparameter fine tuning, I used the following:

training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1

I kept the training batch low because I didn't have a lot of memory storage for both training and evaluation dataset. I only used one epoch to ensure that there was no overfitting.



