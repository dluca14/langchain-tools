# langchain-sandbox

## Installation

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

## Adding packages

```bash
# adding packages from 
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add $PROJECT_NAME

# adding custom GitHub repo packages
langchain app add --repo $OWNER/$REPO
# or with whole git string (supports other git providers):
# langchain app add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
langchain app add $PROJECT_NAME --api_path=/my/custom/path/rag
```

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```

## Setup LangSmith (Optional)
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section


```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```



# Notes

### Fireworks
- Fireworks.ai is a lightning-fast inference platform that helps you serve generative AI models. It hosts fine-tuned models like Mixtral

### LangSmith
- It lets you debug, test, evaluate, and monitor chains and intelligent agents built on any LLM framework and seamlessly integrates with LangChain, the go-to open source framework for building with LLMs. 
- Good for observability over your LLM project.

### Chroma
- Chroma is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0.
- Chroma runs in various modes. See below for examples of each integrated with LangChain. - in-memory - in a python script or jupyter notebook - in-memory with persistance - in a script or notebook and save/load to disk - in a docker container - as a server running your local machine or in the cloud

### Nomic
- Nomic’s nomic-embed-text-v1.5 model was trained with Matryoshka learning to enable variable-length embeddings with a single model. This means that you can specify the dimensionality of the embeddings at inference time. The model supports dimensionality from 64 to 768.

### Docusaurus
- Docusaurus is a static-site generator which provides out-of-the-box documentation features.
- By utilizing the existing SitemapLoader, this loader scans and loads all pages from a given Docusaurus application and returns the main documentation content of each page as a Document.

### Youtube
- https://www.youtube.com/watch?v=Ce03oEotdPs