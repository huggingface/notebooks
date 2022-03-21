# Hugging Face Transformers Amazon SageMaker Examples

Example Jupyter notebooks that demonstrate how to build, train, and deploy [Hugging Face Transformers](https://github.com/huggingface/transformers) using [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) and the [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/).


## 🛠️ Setup


The quickest setup to run example notebooks includes:
- An [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
- Proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup
- An [Amazon SageMaker Notebook Instance](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- An [S3 bucket](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-config-permissions.html)

## 📓 Examples

| Notebook                                                                                                                                                    | Type     | Description                                                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------|
| [01 Getting started with PyTorch](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/sagemaker-notebook.ipynb)       | Training | Getting started end-to-end example on how to fine-tune a pre-trained Hugging Face Transformer for Text-Classification using PyTorch    |
| [02 getting started with TensorFlow](https://github.com/huggingface/notebooks/blob/master/sagemaker/02_getting_started_tensorflow/sagemaker-notebook.ipynb) | Training | Getting started end-to-end example on how to fine-tune a pre-trained Hugging Face Transformer for Text-Classification using TensorFlow |
| [03 Distributed Training: Data Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/03_distributed_training_data_parallelism/sagemaker-notebook.ipynb) | Training | End-to-end example on how to use distributed training with data-parallelism strategy for fine-tuning a pre-trained Hugging Face Transformer for Question-Answering using Amazon SageMaker Data Parallelism |
| [04 Distributed Training: Model Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb) | Training | End-to-end example on how to use distributed training with model-parallelism strategy to pre-trained Hugging Face Transformer using Amazon SageMaker Model Parallelism |
| [05 How to use Spot Instances & Checkpointing](https://github.com/huggingface/notebooks/blob/master/sagemaker/05_spot_instances/sagemaker-notebook.ipynb) | Training | End-to-end example on how to use Spot Instances and Checkpointing to reduce training cost |
| [06 Experiment Tracking with SageMaker Metrics](https://github.com/huggingface/notebooks/tree/master/sagemaker/06_sagemaker_metrics) | Training | End-to-end example on how to use SageMaker metrics to track your experiments and training jobs |
| [07 Distributed Training: Data Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/07_tensorflow_distributed_training_data_parallelism/sagemaker-notebook.ipynb) | Training | End-to-end example on how to use Amazon SageMaker Data Parallelism with TensorFlow |
| [08 Distributed Training: Summarization with T5/BART](https://github.com/huggingface/notebooks/blob/master/sagemaker/08_distributed_summarization_bart_t5/sagemaker-notebook.ipynb) | Training | End-to-end example on how to fine-tune BART/T5 for Summarization using Amazon SageMaker Data Parallelism |
| [09 Vision: Fine-tune ViT](https://github.com/huggingface/notebooks/blob/master/sagemaker/09_image_classification_vision_transformer/sagemaker-notebook.ipynb) | Training | End-to-end example on how to fine-tune Vision Transformer for Image-Classification |
| [10 Deploy HF Transformer from Amazon S3](https://github.com/huggingface/notebooks/blob/master/sagemaker/10_deploy_model_from_s3/deploy_transformer_model_from_s3.ipynb) | Inference | End-to-end example on how to deploy a model from Amazon S3 |
| [11 Deploy HF Transformer from Hugging Face Hub](https://github.com/huggingface/notebooks/blob/master/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb) | Inference | End-to-end example on how to deploy a model from the Hugging Face Hub |
| [12 Batch Processing with Amazon SageMaker Batch Transform](https://github.com/huggingface/notebooks/blob/master/sagemaker/12_batch_transform_inference/sagemaker-notebook.ipynb) | Inference | End-to-end example on how to do batch processing with Amazon SageMaker Batch Transform |
| [13 Autoscaling SageMaker Endpoints](https://github.com/huggingface/notebooks/blob/master/sagemaker/13_deploy_and_autoscaling_transformers/sagemaker-notebook.ipynb) | Inference | End-to-end example on how to do use autoscaling for a HF Endpoint |
| [14 Fine-tune and push to Hub](https://github.com/huggingface/notebooks/blob/master/sagemaker/14_train_and_push_to_hub/sagemaker-notebook.ipynb) | Training | End-to-end example on how to do use the Hugging Face Hub as MLOps backend for saving checkpoints during training |
| [15 Training Compiler](https://github.com/huggingface/notebooks/blob/master/sagemaker/15_training_compiler/sagemaker-notebook.ipynb) | Training | End-to-end example on how to do use Amazon SageMaker Training Compiler to speed up training time |
| [16 Asynchronous Inference](https://github.com/huggingface/notebooks/blob/master/sagemaker/16_async_inference_hf_hub/sagemaker-notebook.ipynb) | Inference | End-to-end example on how to do use Amazon SageMaker Asynchronous Inference endpoints with Hugging Face Transformers |
| [17 Custom inference.py script](https://github.com/huggingface/notebooks/blob/master/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb) | Inference | End-to-end example on how to create a custom inference.py for Sentence Transformers and sentence embeddings |
| [18 AWS Inferentia](https://github.com/huggingface/notebooks/blob/master/sagemaker/18_inferentia_inference/sagemaker-notebook.ipynb) | Inference | End-to-end example on how to AWS Inferentia to speed up inference time |
