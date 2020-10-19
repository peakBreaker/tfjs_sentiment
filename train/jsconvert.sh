#!/usr/bin/env bash

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='SentimentModel' \
    --saved_model_tags=serve \
    ./simple_embeddings_model \
    ../web_model
