from openai.types.chat.chat_completion import ChatCompletion

class TokenCounter:
    def __init__(self):
        self.token_counter = {}
        self.cost_counter = {"prompt": 0, "completion": 0, "total": 0}
        self.token_cost = {
            "gpt-4-1106-preview": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "gpt-4-vision-preview": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "gpt-4-1106-vision-preview": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "gpt-3.5-turbo-1106": {"prompt_tokens": 0.001, "completion_tokens": 0.002},
        }
    
    def update(self, response: ChatCompletion, verbose: bool = False):
        model_name = response.model
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        if verbose:
            print("Latest API Call on {} model usage: Prompt Tokens - {}, Completion Tokens - {}"
                .format(model_name, prompt_tokens, completion_tokens))
        if model_name in self.token_counter:
            self.token_counter[model_name]["prompt_tokens"] += prompt_tokens
            self.token_counter[model_name]["completion_tokens"] += completion_tokens
        else:
            self.token_counter[model_name] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        self.update_cost(prompt_tokens, completion_tokens, model_name, verbose=verbose)
            
    def update_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str = "gpt-4-vision-preview",
                    verbose: bool = False):
        prompt_unit_cost = self.token_cost[model_name]["prompt_tokens"]
        completion_unit_cost = self.token_cost[model_name]["completion_tokens"]
        prompt_cost = prompt_tokens / 1000 * prompt_unit_cost
        completion_cost = completion_tokens / 1000 * completion_unit_cost
        if verbose:
            print("Latest API Call on {} model. Cost: Prompt Cost - {}, Completion Cost - {}"
                .format(model_name, prompt_cost, completion_cost))
        self.cost_counter["prompt"] += prompt_cost
        self.cost_counter["completion"] += completion_cost
        self.cost_counter["total"] += (prompt_cost + completion_cost)
            
    def reset(self):
        self.token_counter = {}
        self.cost_counter = {"prompt": 0, "completion": 0, "total": 0}