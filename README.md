# Intelligent Bot: Development of a State-of-the-Art Question-Answering Model

## Introduction

This repository contains the code and documentation for developing a state-of-the-art question-answering model leveraging the Quora Question Answer Dataset. The objective is to create an AI system capable of understanding and generating accurate responses to a variety of user queries, mimicking human-like interaction.

## Table of Contents

1. [Data Exploration, Cleaning, and Preprocessing](#data-exploration-cleaning-and-preprocessing)
2. [Model Selection and Evaluation](#model-selection-and-evaluation)
3. [Fine-Tuning Models](#fine-tuning-models)
4. [Inference](#inference)
5. [Model Performance](#model-performance)
6. [Insights and Recommendations](#insights-and-recommendations)
7. [Conclusion](#conclusion)

## Data Exploration, Cleaning, and Preprocessing

### Dataset Overview

The dataset used for this project is sourced from the Quora Question Answer Dataset. This dataset contains pairs of questions and answers from Quora, providing a rich source of information for training a question-answering model.

### Data Analysis and Cleaning

- **Structure and Content Analysis**: Analyzed the dataset to understand its structure and content.
- **Data Cleaning**: Removed any irrelevant information, including duplicates, null values, non-English content, HTML URLs, and common stop words. Also converted emojis to text meanings, removed punctuation marks, and corrected spelling errors.
- **Preprocessing Techniques**:
  - Tokenization
  - Stop Word Removal
  - Stemming/Lemmatization
  - Punctuation Removal
  - Emoji Conversion
  - Spell Correction

### Inference on Data Changes

The average word length of questions was initially around 30 and reduced to 20 after preprocessing. For answers, it was initially around 250 and reduced to 150.

## Model Selection and Evaluation

### Model Testing

Various NLP models were tested to determine the best performance for our question-answering task. The models considered include:

- **FLAN-T5**: Fine-tuned using the Flan dataset to improve its performance on various NLP tasks.
- **LLaMA3**: Designed to handle complex language understanding tasks.
- **Mistral7B**: A robust NLP model designed for high-performance language tasks.

### Evaluation Metrics

The models were evaluated using the following metrics:

- **ROUGE-1**: Measures the overlap of unigrams between the predicted and reference answers.
- **ROUGE-2**: Measures the overlap of bigrams between the predicted and reference answers.
- **ROUGE-L**: Measures the longest common subsequence between the predicted and reference answers.

## Fine-Tuning Models

### FLAN-T5 Fine-Tuning

The FLAN-T5 model was fine-tuned on the Quora dataset using the Hugging Face transformers library. The training process involved several key steps to adjust the model parameters to better understand and generate answers based on the Quora question-answer pairs.

### LLaMA3 Fine-Tuning

The LLaMA3 model was fine-tuned on the Quora dataset using a similar approach, with adjustments made for the specific architecture and requirements of the LLaMA3 model.

### Mistral7B Fine-Tuning

The Mistral7B model was fine-tuned using a tailored approach to leverage its extensive pre-training and enhance its question-answering capabilities.

## Inference

### FLAN-T5 Inference

After fine-tuning, the FLAN-T5 model was used for inference to generate answers to user queries. The process involved loading the fine-tuned model and tokenizer, encoding the input questions, generating answers, and decoding the outputs.

### LLaMA3 Inference

The LLaMA3 model was similarly used for inference, with specific adjustments for its architecture.

### Mistral7B Inference

The Mistral7B model's inference process involved loading the model and tokenizer, encoding the questions, generating answers, and decoding the outputs to produce human-readable responses.

## Model Performance

### Graphical Representation of Model Performance

Graphs and charts were generated to illustrate the performance of different models based on the evaluation metrics. The metrics used to evaluate the models were ROUGE-1, ROUGE-2, and ROUGE-L scores.

## Insights and Recommendations

### Insights

- **Data Insights**: The dataset contains a wide variety of question types and answer patterns, which helps in training a robust question-answering model.
- **Model Performance**: The FLAN-T5, LLaMA3, and Mistral7B models showed different strengths, with Mistral7B performing the best overall in terms of ROUGE scores.

### Recommendations

1. **Model Enhancements**:
   - **Fine-tuning Hyperparameters**: Further fine-tuning of hyperparameters such as learning rate, batch size, and number of training epochs could improve model performance.
   - **Additional Training Data**: Incorporating more diverse and extensive training data could help the model learn better and improve its generalization capabilities.

2. **Future Research**:
   - **Ensemble Methods**: Exploring ensemble methods, where multiple models are combined to generate answers, could yield better results by leveraging the strengths of each individual model.
   - **Other NLP Models and Techniques**: Investigating other advanced NLP models and techniques, such as transformer-based models with more parameters or different architectures, could provide additional performance improvements.

3. **Customizing Models for InterGlobe Aviation Ltd (GoIndiGo)**:
   - **Leveraging LangChain Techniques**: To build an in-house AI for specific domains such as aviation, we can leverage LangChain techniques to customize our fine-tuned models for aviation datasets of InterGlobe Aviation Ltd (GoIndiGo). LangChain provides tools and methodologies to create chains of language models that can be tailored to specific tasks and domains, ensuring that the AI system is highly specialized and effective in the target area.
   - **Domain-Specific Training**: By fine-tuning the models on aviation-specific datasets, the AI can learn the terminology, context, and nuances of the aviation industry, resulting in more accurate and relevant responses.
   - **Integration with Existing Systems**: LangChain techniques allow seamless integration with existing systems and workflows at GoIndiGo, making it easier to deploy the AI models in a real-world environment. This can streamline operations and enhance the efficiency of various processes within the aviation sector.
   - **Custom Workflows and Pipelines**: Creating custom workflows and pipelines tailored to the specific needs of GoIndiGo ensures that the AI system can handle a variety of tasks with high accuracy and reliability.

## Conclusion

This repository presents a comprehensive approach to developing a question-answering model using the Quora Question Answer Dataset. The steps taken include data exploration, model selection, evaluation, fine-tuning, and inference, leading to meaningful insights and recommendations.

## Acknowledgment of Resources
The fine-tuning and inference processes for this project were performed using Kaggle's free limited GPU resources, specifically the NVIDIA P100 GPUs. These resources provided the necessary computational power to train and evaluate the models effectively within the constraints of a limited budget.

## Important Links

**Training Report**
-[Flan T5](https://api.wandb.ai/links/amazeml/1l8yu08q)
-[Mistral 7B](https://api.wandb.ai/links/amazeml/b1f768wu)
-[Llama3 8B](https://api.wandb.ai/links/feluda0307-gojo-squad/oqm6c9dr)

**Fine Tuned Models Hugging Face Link**
-[Flan T5](https://huggingface.co/Feluda/Flan-t5-FineTuned-Quora)
-[Mistral 7B](https://huggingface.co/Feluda/mistral_fine_tune)
-[Llama3 8B](https://huggingface.co/Feluda/llama-3-8b-chat-Qs-Ans-Quora)
Feel free to clone the repository and explore the code and documentation to understand the process and replicate the results. For any questions or suggestions, please open an issue or contact me directly.
