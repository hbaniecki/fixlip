from copy import deepcopy

import numpy as np

try:
    import clip
except ImportError:
    print("Warning: no clip module found")
import torch

from shapiq import Game


class CLIPGame(Game):
    """
    A custom interface for OpenAI CLIP
    """
    def __init__(self, model, processor, input_image, input_text, patch_size, batch_size=1, verbose=False):
        self.model = model
        self.processor = processor
        self.input_image = input_image
        self.input_text = input_text
        self.batch_size = batch_size

        self.inputs = self._processor_function([input_image], [input_text])

        self.image_size = model.visual.input_resolution
        self.patch_size = patch_size
        self.n_channels = 3
        self.grid_size = self.image_size // self.patch_size
        self.n_players_image = int(self.image_size / self.patch_size) ** 2 
        # remove the bos and eos tokens
        self.n_players_text = self.inputs[1].count_nonzero().item() - 2 
        self.text_context_length = model.context_length
        self.device = next(model.parameters()).device

        # get the normalization value
        coalitions = np.zeros((2, self.n_players_image + self.n_players_text), dtype=bool)
        coalitions[1, :] = True
        game_output = self.value_function(coalitions=coalitions)
        self.empty_value = float(game_output[0])
        self.full_value = float(game_output[1])

        if verbose:
            print(f"Similarly of the Image and Text: {self.full_value} (empty_value={self.empty_value})")


        super().__init__(
            n_players=self.n_players_image + self.n_players_text,
            normalize=True,
            normalization_value=self.empty_value,
            verbose=False
        )


    def _processor_function(self, input_image, input_text):
        """
        Input: list of images of length N, list of texts of length M.
        Output: a list of processed inputs as ['pixel_values', 'input_ids']
        """
        text = clip.tokenize(input_text)
        image = torch.stack([self.processor(i) for i in input_image])
        return [image, text]


    def value_function(self, coalitions, batch_size=None):
        """ Baseline value function
        Input: Coalitions of the game as a boolean np.array of shape (n_coalitions, n_players).
        Output: Model outputs for the coalitions of shape (n_coalitions, )."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions = coalitions.shape[0]
        coalitions_image = torch.from_numpy(coalitions[:, :self.n_players_image])
        coalitions_text = torch.from_numpy(coalitions[:, self.n_players_image:])
        
        # [n_coallitions, text_context_length]
        text_binary_masks = torch.cat(
            (torch.ones(n_coalitions, 1), coalitions_text, 
             torch.ones(n_coalitions, 1), torch.zeros(n_coalitions, self.text_context_length - self.n_players_text - 2)), 
            axis=1
        ).int()
        # [n_coallitions, n_channels, image_size, image_size]
        image_binary_masks = self._generate_image_binary_mask(coalitions_image)
        
        inputs_original = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_size)

        #:# batch processing
        batch_iters = n_coalitions // batch_size
        batch_left = n_coalitions % batch_size
        coalitions_outputs = []
        for batch_index in range(batch_iters + 1):
            if batch_index < batch_iters:
                inputs = deepcopy(inputs_original)
                inputs[0] = (inputs[0] * image_binary_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)]).to(self.device)
                inputs[1] = (inputs[1] * text_binary_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)] +\
                             289 * (1 - text_binary_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)])).to(self.device)
            elif batch_left > 0: # process last batch (once)
                inputs = self._processor_function([self.input_image]*batch_left, [self.input_text]*batch_left)
                inputs[0] = (inputs[0] * image_binary_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)]).to(self.device)
                inputs[1] = (inputs[1] * text_binary_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)] +\
                             289 * (1 - text_binary_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)])).to(self.device)
            else:
                break 
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(*inputs)
            # take only the diagonal predictions - a naive approach
            outputs = torch.diagonal(logits_per_image).cpu()
            coalitions_outputs.append(outputs)
        coalitions_outputs = torch.concat(coalitions_outputs)
        return coalitions_outputs.numpy()


    def value_function_crossmodal(self, coalitions_image, coalitions_text, batch_size=None):
        """ Efficient value function
        Input: Coalitions of the game as two boolean np.arrays of shapes 
            (n_coalitions_image, n_players_image) and (n_coalitions_text, n_players_text).
        Output: Model outputs for the coalitions of shape (n_coalitions_image, n_coalitions_text)."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions_image = coalitions_image.shape[0]
        n_coalitions_text = coalitions_text.shape[0]

        # [n_coallitions, text_context_length]
        text_binary_masks = torch.cat(
            (torch.ones(n_coalitions_text, 1), torch.from_numpy(coalitions_text), 
             torch.ones(n_coalitions_text, 1), torch.zeros(n_coalitions_text, self.text_context_length - self.n_players_text - 2)), 
            axis=1
        ).int()
        # [n_coallitions, n_channels, image_size, image_size]
        image_binary_masks = self._generate_image_binary_mask(torch.from_numpy(coalitions_image))

        inputs_original = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_size)

        #:# batch processing
        batch_iters_image = n_coalitions_image // batch_size
        batch_iters_text = n_coalitions_text // batch_size
        batch_left_image = n_coalitions_image % batch_size
        batch_left_text = n_coalitions_text % batch_size
        if batch_left_text > 0: # to be copied in (batch_iters_image - 1) iterations
            inputs_left_text = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_left_text)

        coalitions_outputs = []
        for batch_index_image in range(batch_iters_image + 1):
            coalitions_outputs_image = []
            if batch_index_image < batch_iters_image:
                inputs_image = deepcopy(inputs_original)
                inputs_image[0] = (inputs_image[0] *\
                              image_binary_masks[(batch_index_image * batch_size):((batch_index_image + 1) * batch_size)]).to(self.device)
            elif batch_left_image > 0: # process last image batch (once)
                inputs_image = self._processor_function([self.input_image] * batch_left_image, [self.input_text] * batch_size)
                inputs_image[0] = (inputs_image[0] *\
                              image_binary_masks[(batch_index_image * batch_size):(batch_index_image * batch_size + batch_left_image)]).to(self.device)
            else:
                break 
            for batch_index_text in range(batch_iters_text + 1):
                if batch_index_text < batch_iters_text:
                    inputs = deepcopy(inputs_image)
                    inputs[1] = (inputs[1] *\
                                  text_binary_masks[(batch_index_text * batch_size):((batch_index_text + 1) * batch_size)] +\
                                    289 * (1 - text_binary_masks[(batch_index_text * batch_size):((batch_index_text + 1) * batch_size)])).to(self.device)
                elif batch_left_text > 0 and batch_index_image < batch_iters_image: # process last text batch in non-terminal image batch
                    inputs = deepcopy(inputs_left_text)
                    inputs[0] = (inputs[0] *\
                                image_binary_masks[(batch_index_image * batch_size):((batch_index_image + 1) * batch_size)]).to(self.device)
                    inputs[1] = (inputs[1] *\
                                  text_binary_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)] +\
                                    289 * (1 - text_binary_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)])).to(self.device)                
                elif batch_left_text > 0 and batch_left_image > 0: # process last text and image batch (once)
                    inputs = self._processor_function([self.input_image] * batch_left_image, [self.input_text] * batch_left_text)
                    inputs[0] = (inputs[0] *\
                                image_binary_masks[(batch_index_image * batch_size):(batch_index_image * batch_size + batch_left_image)]).to(self.device)
                    inputs[1] = (inputs[1] *\
                                  text_binary_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)] +\
                                    289 * (1 - text_binary_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)])).to(self.device)            
                else:
                    break
                with torch.no_grad():
                    logits_per_image, logits_per_text = self.model(*inputs)
                outputs = logits_per_image.cpu()
                coalitions_outputs_image.append(outputs)
            coalitions_outputs.append(torch.concat(coalitions_outputs_image, axis=1))
        coalitions_outputs = torch.concat(coalitions_outputs, axis=0)

        return coalitions_outputs.numpy()
    

    #:# ---------- utility functions ---------- #:#

    def _generate_image_binary_mask(self, coalitions):
        """
        Input: binary torch tensor
        Output: binary torch tensor
        """
        n_coalitions = coalitions.shape[0]
        # Expand each coalition value into a patch
        binary_masks = coalitions\
            .repeat_interleave(self.patch_size**2, dim=1)\
                .reshape(n_coalitions, self.grid_size, self.grid_size, self.patch_size, self.patch_size)
        # Rearrange to form the final batch of full-size images
        binary_masks = binary_masks\
            .permute(0, 1, 3, 2, 4)\
                .reshape(n_coalitions, self.image_size, self.image_size)
        # Add image channel dimension
        binary_masks = binary_masks\
            .repeat((self.n_channels, 1, 1, 1))\
                .permute(1, 0, 2, 3)
        return binary_masks
    


#:# Find mask token 289
# https://github.com/openai/CLIP/issues/14#issuecomment-763839310
# import regex as re
# from clip.simple_tokenizer import SimpleTokenizer
# _tokenizer = SimpleTokenizer()
# print([_tokenizer.decode([i])[0] for i in [0, 288]])
# print([_tokenizer.encode(_tokenizer.decode([i])[0]) for i in [0, 288]])
# bpe_tokens = []
# text = "a A b B"
# for token in re.findall(re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE), text):
#     token = ''.join(_tokenizer.byte_encoder[b] for b in token.encode('utf-8'))
#     bpe_tokens.extend(_tokenizer.encoder[bpe_token] for bpe_token in _tokenizer.bpe(token).split(' '))
# bpe_tokens