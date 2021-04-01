#coding=utf8
import torch
import torch.nn as nn
from models.nlg.nlg_model import NLGModel

class SCLSTMModel(NLGModel):

    def forward(self, intents, slots, slot_lens, lens, dec_inputs, slot_states, copy_tokens):
        intents = self.intent_embed(intents)
        memory = self.encoder(self.word_embed(slots), slot_lens, lens)
        decoder_state = self.enc2dec(memory, slot_lens, intents)
        inputs = self.word_embed(dec_inputs)
        decoder_state = (decoder_state[0].contiguous().transpose(0, 1), decoder_state[1].contiguous().transpose(0, 1))
        feats, _, slots = self.decoder(inputs, intents, memory, decoder_state, slot_states, slot_lens, all_slot_hist=True)
        log_prob = self.generator(feats)
        return log_prob, slots

    def decode_one_step(self, ys, intents, memory, dec_states, slot_states, slot_lens, copy_tokens):
        inputs = self.word_embed(ys)
        feats, dec_states, slot_states = self.decoder(inputs, intents, memory, dec_states, slot_states, slot_lens)
        logprob = self.generator(feats).squeeze(dim=1) # bsize x vocab_size
        return logprob, (dec_states, slot_states)
