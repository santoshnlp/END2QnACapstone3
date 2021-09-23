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
The dataset should contain input to encoder and decoder.  

==============================
BART encoder input
==============================

This is concatenate text having  1. question  2. Top 3 context found by FAISS.

For the purpose of training teacher forcing was used.  If FAISS model failed to provide  the correct document in top3,  one of document was replaced by correct document.

==============================
BART decoder input
==============================

This the actual answer for the given question as per the document.


