#!/bin/bash

conda create -y -p ./env python=3.12

rm .env
touch .env

# Prompt user for API key
echo "Enter your Hugging Face API key:"
read -r api_key
echo "HF_API_KEY=$api_key" >> .env

echo "Enter your AWS public key:"
read -r api_key
echo "AWS_ACCESS_KEY=$api_key" >> .env

echo "Enter your AWS secret key:"
read -r api_key
echo "AWS_SECRET_KEY=$api_key" >> .env

# Set appropriate permissions (readable by owner only)
chmod 600 .env

