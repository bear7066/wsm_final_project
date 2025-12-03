## 說明
1. this branch have new components: expander, llm_based reranker and english prompt
2. baseline2 can now execute evaluation and checking format by ```run.sh, check_format.sh```
3. you can change evaluation method by modify the rageval version in ```run.sh```


### Prerequisites

1. ```pip install -r requirements.txt```
2. ```ollama serve```
3. ```ollama pull llama3.3:70b```

> I use python 3.10.13

### Execution 
1. ``` sh run.sh``` or ``` ./run.sh```, complete path of run.sh should be set by yourself.

### Implementation
1. git branch << name of the branch >>


### To Do
1. Figure out the training dataset of the ```granite4:3b```. We can build our system customizely with the same language.
2. Figure out where to use embedding models: ```embeddinggemma: 300m```, ```qwen3-embedding:0.6b```

### Submission(discuss before submission)

1. copy the ```submit_template.yml``` and 
2. steps are on the slide ```tutorial3```
3. daily limit: 3 times
4. time limit: 3 hours
5. package size: 5GB (can use other model but not exceed it)
6. ollama host should be replace to: http://ollama-gateway:11434
