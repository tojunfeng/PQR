class Prompter:

    def __init__(self, dataset):
        self.dataset = dataset

    def prompt_fn(self, passage):
        if self.dataset == "nfcorpus":
            prompt = f'''{passage}
Please read the above passage about medical information, and write a Question from a different perspective that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''

        elif self.dataset == "scifact":
            prompt = f'''{passage}
Please read the above Passage about scientific claim, and write a Question from a different perspective that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''
        
        elif self.dataset == "scidocs":
            prompt = f'''{passage}
Please read the above Passage about scientific claim, and write a Question from a different perspective that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''
        
        elif self.dataset == "arguana":
            prompt = f'''{passage}
Please read the above Passage, and write a Question from a different perspective that a dense retrieval model can use to find this passage. Directly output the Question without any additional information.
Question: '''
            
        elif self.dataset == "fiqa":
            prompt = f'''{passage}
Please read the above Passage about financial information, and write a Question from a different perspective that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''
            
        elif self.dataset == "trec-covid":
            prompt = f'''{passage}
Please read the above Passage about biomedical information, and write a Question from a different perspective that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''
            
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset}")

        return prompt



class PrompterCot:
    def __init__(self, dataset):
        self.dataset = dataset

    def topic_prompt(self, passage):
        prompt = f'''{passage}
Please read the above Passage and summarize a Topic it includes. Output the Topic directly without any additional information.
Topic: '''
        return prompt
    
    def cot_prompt(self, passage, topic):
        if self.dataset == "nfcorpus":
            prompt = f'''{passage}
Please read the above Passage about medical information, and write a Question related to \"{topic}\" from from a different perspective that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''

        elif self.dataset == "scifact":
            prompt = f'''{passage}
Please read the above Passage about scientific claim, and write a Question related to \"{topic}\" that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''
        
        elif self.dataset == "scidocs":
            prompt = f'''{passage}
Please read the above Passage about scientific claim, and write a Question related to \"{topic}\" that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''
        
        elif self.dataset == "arguana":
            prompt = f'''{passage}
Please read the above Passage, and write a Question related to \"{topic}\" that a dense retrieval model can use to find this Passage. Directly output the Question without any additional information.
Question: '''
            
        elif self.dataset == "fiqa":
            prompt = f'''{passage}
Please read the above Passage about financial information, and write a Question related to \"{topic}\" that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''
            
        elif self.dataset == "trec-covid":
            prompt = f'''{passage}
Please read the above Passage about biomedical information, and write a Question related to \"{topic}\" that dense retrieval model could use to find this Passage. Directly output the Question without any additional information.
Question: '''

        else:
            raise ValueError(f"Invalid dataset name: {self.dataset}")

        return prompt
