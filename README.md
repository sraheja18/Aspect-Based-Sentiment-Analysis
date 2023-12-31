# Aspect-Based-Sentiment-Analysis

Welcome to the Aspect-Based Sentiment Analysis project! This repository is your gateway to understanding and implementing one of the most powerful techniques
in natural language processing (NLP) - Aspect-Based Sentiment Analysis (ABSA).

## Table of Contents
- Introduction to ABSA
- Key Features
- Getting Started
- Installation
- Usage
- Examples


## 1. Introduction to ABSA
Aspect-Based Sentiment Analysis is a subfield of NLP that focuses on extracting and analyzing the sentiment expressed towards specific aspects or features of a product, service,
or entity mentioned in text data. It enables a deeper understanding of not just whether a piece of text is positive or negative, but also which particular aspects or entities within that text are being referred to and how sentiments are associated with them.

ABSA is invaluable in various domains, including customer reviews analysis, market research, product development, and customer support. It allows businesses and organizations to gain 
fine-grained insights into what customers like or dislike about their offerings, leading to informed decision-making and enhanced customer satisfaction.

## 2. Key Features
In this repository, you will find resources and code that enable you to:

- Perform Aspect-Based Sentiment Analysis on text data.
- Extract and classify aspects or entities mentioned in text.
- Determine the sentiment (e.g., positive, negative, neutral) associated with each aspect/entity.

## 3. Getting Started
Before diving into the code, it's essential to understand the components of ABSA:

- Aspect Extraction: Identifying and extracting aspects or entities from the text. For instance, in a product review, aspects could be "battery life," "camera quality," and "build quality."
- Sentiment Classification: Determining the sentiment polarity (positive, negative, neutral) associated with each extracted aspect or entity.

## 4. Installation
To get started with this project, you'll need to install the necessary libraries and dependencies. Check the requirements file and install the libraries using pip install -r requirements.txt

## 5. Usage
Two models need to be trained. One for aspect term extraction and the other for aspect based sentiment analysis. They can be trained by running the files train_ATE.py and train_ABSA.py. For inference, refer to the notebook inference_combined.ipynb.

## 6. Examples
- Sentence: For the price you pay, this product is very good. However, battery life is a little lack-luster coming from a MacBook Pro. <br />
Aspects: ['price', 'battery life'] <br />
Term: ['price'] , Class: Positive , Probability: 0.914 <br />
Term: ['battery life'] , Class: Negative , Probability: 0.999 <br />

- Sentence: Speakers are great but screen colors are dull. <br />
Aspects: ['speakers', 'screen colors'] <br />
Term: ['speakers'] , Class: Positive , Probability: 0.999 <br />
Term: ['screen colors'] , Class: Negative , Probability: 1.0 <br />
## 7.  Here's the final webpage
![Screenshot](ABSA_webpage.png)

## 6. Credits - https://github.com/1tangerine1day/Aspect-Term-Extraction-and-Analysis
