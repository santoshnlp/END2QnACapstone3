##################
Table of Contents
##################
.. contents::
  :local:
  :depth: 4
  
***************
Model
***************
We will be using BART model.  Given the context and question, it should be able to give us answer.


==============================
BART
==============================

In this project we explain the sequence to sequence modeling using [`HuggingFace <https://huggingface.co/transformers/model_doc/bart.html>`_].

.. code-block:: python

 from transformers import BartTokenizer, BartForConditionalGeneration
 model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')



***************
Dataset Preparation
***************





