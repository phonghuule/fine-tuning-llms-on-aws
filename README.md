# Fine-tuning LLMs using Amazon SageMaker, AWS Trainium, and Optimum Neuron

This lab is provided as part of **[AWS Innovate AI/ML and Data Edition](https://aws.amazon.com/events/aws-innovate/apj/aiml-data/)**.

ℹ️ You will run this lab in your own AWS account. Please follow directions at the end of the lab to remove resources to avoid future costs.

ℹ️ Please let us know what you thought of this session and how we can improve the experience for you in the future by completing [the survey](#survey) at the end of the lab.
Participants who complete the surveys from AWS Innovate Online Conference will receive a gift code for USD25 in AWS credits <sup> 1, 2 & 3 </sup>.

## **Overview**
In this lab, you will learn how to use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) to fine-tune a pretrained Hugging Face LLM using [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) accelerators, and then leverage the fine-tuned model for inference on [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/). 

This is a labified version of this [AWS Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/0cd5851b-5253-4a65-b351-70d0d80a7fb5/en-US)

## Setup
[Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html) is a web-based, integrated development environment (IDE) for machine learning that lets you build, train, debug, deploy, and monitor your machine learning models. Studio provides all the tools you need to take your models from experimentation to production while boosting your productivity. In addition, since all these steps of the ML workflow are tracked within the environment, you can quickly move back and forth between steps, and also clone, tweak and replay them. This gives you the ability to make changes quickly, observe outcomes, collaborate effectively and iterate faster, reducing the time to market for your high quality ML solutions.

It is recommended to run this workshop from the **us-west-2** region. If you already have SageMaker Studio configured in **us-west-2**, please skip ahead to Step 10 to download the notebooks and related files.

1. Open the [AWS console](https://console.aws.amazon.com/) and make sure to select **US West (Oregon) / us-west-2** using the dropdown menu at the upper-right of the console:

![Choose Region](/images/choose_region.png)

2. In the search bar, type **SageMaker** and click on **Amazon SageMaker**.

![SageMaker Dropdown](/images/sagemaker_dropdown.png)

3. From the left-hand menu, choose **Studio** and then click the **Create a SageMaker domain** button at the right side of the screen.

![SageMaker Domain](/images/sagemaker_domain.png)

4. Choose **Set up for single user (Quick setup)** and then click the **Set up** button to begin creating your SageMaker domain.

![Quick Setup](/images/quick_setup.png)

5. Wait a few minutes while the SageMaker domain is being created. Once the domain has been created, click the Launch dropdown and choose Studio.

![Launch Domain](/images/launch_domain.png)

6. When the SageMaker Studio landing page opens, choose Studio Classic from the top-left Applications pane.

![Studio Landing Page](/images/sm_studio_landing_page.png)

7. You will be redirected to your Studio environment that will look similar to the screen below.

![Studio Environment](/images/sm_studio_env.png)

8. We now need to download the workshop notebooks and related files to your SageMaker Studio environment. Begin by choosing **File -> New -> Terminal** from the topmost menu in SageMaker Studio

![Launch Terminal](/images/launch_terminal.png)

9. Inside the terminal, run the following sudo and wget commands to download the notebooks and related files:

```
sudo yum install wget -y

wget -P 01_finetuning https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/0cd5851b-5253-4a65-b351-70d0d80a7fb5/01_finetuning/optimum_neuron-0.0.14.dev0-py3-none-any.whl

wget -P 01_finetuning https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/0cd5851b-5253-4a65-b351-70d0d80a7fb5/01_finetuning/requirements.txt

wget -P 01_finetuning https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/0cd5851b-5253-4a65-b351-70d0d80a7fb5/01_finetuning/run_clm.py

wget -P 01_finetuning https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/0cd5851b-5253-4a65-b351-70d0d80a7fb5/01_finetuning/Finetune-TinyLlama-1.1B.ipynb

wget -P 02_inference https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/0cd5851b-5253-4a65-b351-70d0d80a7fb5/02_inference/Inference-TinyLlama-1.1B.ipynb
```

10. Navigate to the notebook folder **/01_finetuning/** and launch **Finetune-TinyLlama-1.1B.ipynb** notebook.

![Start Notebook](/images/start_notebook.png)

11. Set up Python environment for the notebook.
Before we begin training, you need to set up the notebook environment in your SageMaker. Once you started a **SageMaker Studio**, select the Image **Data Science 3.0** with **Python 3** kernel and **ml.t3.medium** instance type as per the image below:

ℹ️ The instance type here is specifically intended for running SageMaker notebook and is not designed for running training job or deploying the model.

![Notebook Environment](/images/notebook_env.png)

## Lab
### Fine-tuning an LLM
In this section we will use Amazon SageMaker to run an LLM fine-tuning job using the Hugging Face Optimum Neuron library and an AWS EC2 trn1.2xlarge instance (featuring Trainium accelerators).

ℹ️  Navigate to the notebook folder /01_finetuning/ and launch the Finetune-TinyLlama-1.1B.ipynb notebook.

Please refer to the above-mentioned Jupyter notebook and run each of the notebook cells in order to complete this lab module. The following content is provided as reference material which may help you as you work through the notebook.

#### AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) is the second-generation machine learning (ML) accelerator that AWS purpose built for deep learning training of 100B+ parameter models. Each Amazon Elastic Compute Cloud (EC2) Trn1 instance deploys up to 16 AWS Trainium accelerators to deliver a high-performance, low-cost solution for deep learning (DL) training in the cloud. Although the use of deep learning is accelerating, many development teams are limited by fixed budgets, which puts a cap on the scope and frequency of training needed to improve their models and applications. Trainium based EC2 Trn1 instances solve this challenge by delivering faster time to train while offering up to 50% cost-to-train savings over comparable Amazon EC2 instances.

In this lab you will work with a single trn1.2xlarge instance containing a single Trainium accelerator with 2 NeuronCores. While the trn1.2xlarge instance is useful for fine-tuning, AWS also offers larger trn1.32xlarge and trn1n.32xlarge instance types that each contain 16 Trainium accelerators (32 NeuronCores) and are capable of large-scale distributed training.

#### Hugging Face Optimum Neuron
[Optimum Neuron](https://huggingface.co/docs/optimum-neuron) is a Python library providing the interface between Hugging Face Transformers and the purpose-built AWS ML accelerators - Trainium and Inferentia. Optimum Neuron provides a set of tools enabling easy model loading, training, and inference on single and multi-accelerator settings for different downstream tasks. The goal of this library is to provide an easy-to-use mechanism for users to leverage the most popular Hugging Face models on Trainium and Inferentia, using familiar Transformers concepts such as the Trainer API. 

### Deploying a fine-tuned LLM for inference
In this section we will use Amazon SageMaker and Amazon EC２ Inf２ instance to deploy the model fine-tuned in the previous section. Amazon SageMaker deployment provides fully managed options for deploying our models using Real Time or Batch modes. AWS Inferentia gives best cost per inference.

ℹ️   Navigate to the notebook folder /02_inference/ and launch Inference-TinyLlama-1.1B.ipynb notebook.

Please refer to the above-mentioned Jupyter notebook and run each of the notebook cells in order to complete this lab module. The following content is provided as reference material which may help you as you work through the notebook.

#### Inferentia2
[AWS Inferentia2](https://aws.amazon.com/machine-learning/inferentia) is the second generation purpose built Machine Learning inference accelerator from AWS. Amazon EC2 Inf2 instances are powered by AWS Inferentia2. Inf2 instances raise the performance of Inf1 by delivering 3x higher compute performance, 4x larger total accelerator memory, up to 4x higher throughput, and up to 10x lower latency. Inf2 instances are the first inference-optimized instances in Amazon EC2 to support scale-out distributed inference with ultra-high-speed connectivity between accelerators. You can now efficiently and cost-effectively deploy models with hundreds of billions of parameters across multiple accelerators on Inf2 instances.

In this lab you will work with a single inf2.xlarge instance containing a single Inferentia2 accelerator with 2 NeuronCores. While the inf2.xlarge instance is useful for most cost-optimized inference, AWS also offers larger inf2.48xlarge and inf2.24xlarge instance types that each contain 12 and 6 Inferentia2 accelerators (24 and 12 NeuronCores) and are capable of more performance optimized inference. 

## Summary
Now that you have completed the session, you should have an understanding of:
- Using Optimum Neuron to streamline the fine-tuning of LLMs with AWS Trainium
- Launching an LLM fine-tuning job using SageMaker Training and storing your fine-tuned model in Amazon S3
- Preparing your model for deployment on AWS Inferentia using transformers-neuronx
- Deploying your model on a SageMaker Hosting endpoint using DJL Serving

## **Clean Up**
1. Return to the AWS Console, and search for Amazon SageMaker to launch the SageMaker console. In the left-hand menu, choose **Training -> Training jobs**. If any training jobs show a status of Running, please select the jobs and choose **Action -> Stop**to stop the jobs. Repeat this step for any additional regions that you used during the lab.

2. Choose **Inference -> Inference endpoints** from the left-hand menu. If any endpoints appear in the list (they should have tinyllama in the name), please select and delete the endpoints using the Actions menu. Repeat this step for any additional regions that you used during the workshop.

3. From the left-hand menu in the SageMaker console, choose **Domains**. Select the domain that you created for this workshop (likely starts with **QuickSetupDomain-**). On the **User Profiles** screen, select the user profile to be deleted (ex: default-20231211t230805). On the Apps screen, select the **'default'** app and scroll to the right to expose an Action menu. From the Action menu, choose Delete. When the confirmation appears, choose Yes, delete app and then type 'delete' in the textbox to to delete the user profile.

4. Once the SageMaker Studio user profile has been deleted, return to the SageMaker console and choose **Domains** from the left-hand menu. Select the radio button next to the domain you created for this lab, and then click Edit. At the top of the **General Settings** screen you should see a **Delete domain** button. Click this button and follow the prompts to remove the SageMaker Studio domain from your account.

5. At the top of the AWS Console, search for S3 to open the S3 console. In the list of S3 buckets, look for buckets that start with **sagemaker-**. View the contents of these buckets and delete any artifacts that you do not want to keep.

## Survey
Let us know what you thought of this session and how we can improve the presentation experience for you in the future by completing [this event session poll](https://amazonmr.au1.qualtrics.com/jfe/form/SV_5BWPHDlxVcsRbo2?Session=HOL02). 
Participants who complete the surveys from AWS Innovate Online Conference will receive a gift code for USD25 in AWS credits <sup> 1, 2 & 3 </sup>. AWS credits will be sent via email by November 30, 2023.
Note: Only registrants of AWS Innovate Online Conference who complete the surveys will receive a gift code for USD25 in AWS credits via email.

<sup>1</sup>AWS Promotional Credits Terms and conditions apply: https://aws.amazon.com/awscredits/

<sup>2</sup>Limited to 1 x USD25 AWS credits per participant.

<sup>3</sup>Participants will be required to provide their business email addresses to receive the gift code for AWS credits.