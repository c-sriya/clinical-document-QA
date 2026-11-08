# Clinical Document Q&A System (CRAG + Eval Suite)

A corrective-RAG (CRAG) pipeline for answering clinical questions over PubMed
abstracts, built incrementally from a naive RAG baseline and measured at each
stage with RAGAS.

## Status
In development — building incrementally (naive RAG → hybrid retrieval → CRAG → citations).

## Overview
This project retrieves relevant PubMed abstracts to answer clinical questions,
using an LLM to grade retrieval quality and correct poor retrievals before
generating an answer (Corrective RAG). Performance is tracked across pipeline
versions using RAGAS metrics.
