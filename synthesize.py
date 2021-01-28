import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
import random
from string import punctuation
from g2p_en import G2p

from fastspeech2 import FastSpeech2
from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
import audio as Audio
from utils import get_mask_from_lengths
import soundfile

#import the modules needed for fine-tuning
from torch.utils.data import DataLoader
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hp.use_spk_embed:
    if hp.dataset == "VCTK":
        from data import vctk
        spk_table, inv_spk_table = vctk.get_spk_table()
        
    if hp.dataset == "LibriTTS":
        from data import libritts
        spk_table, inv_spk_table = libritts.get_spk_table()

def preprocess(text):        
    text = text.rstrip(punctuation)

    g2p = G2p()
    phone = g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{'+ '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')
    print(text)
    print('|' + phone + '|')
    print("\n")
    #print(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)

#def get_FastSpeech2(num):
def get_FastSpeech2(num, loader):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    n_spkers = torch.load(checkpoint_path)['model']['module.embed_speakers.weight'].shape[0]
    
    if hp.use_spk_embed:    
        #model = nn.DataParallel(FastSpeech2(True, n_spkers)).to(device)
        model = FastSpeech2(True, n_spkers).to(device)
    else:
        model = nn.DataParallel(FastSpeech2()).to(device)
    #model.load_state_dict(torch.load(checkpoint_path)['model'])
    checkpoint = torch.load(checkpoint_path)
    for n, p in checkpoint['model'].items():
        if n[7:] not in model.state_dict():
            print('not in meta_model:', n)
            continue
        if isinstance(p, nn.parameter.Parameter):
            p = p.data
        model.state_dict()[n[7:]].copy_(p)


    #################
    #fine-tuning
    #optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.step)
    Loss = FastSpeech2Loss().to(device)

    #fine-tuning
    print('start fine-tuning')
    model = model.train()
    #check grad
    for n,p in model.named_parameters():
        #if n[7:10] not in {'var', 'dec', 'enc', 'pos', 'mel', 'enc'} or n[15:18]=='pos':
        if n[:3] not in {'var'}:
            p.requires_grad = False
        print(n, p.requires_grad)
    current_step = 0
    print('while loop')
    while current_step < hp.syn_fine_tune_step:
        for i,batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                # Get Data
                text = torch.from_numpy(data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

                spk_ids = torch.tensor([7]*hp.batch_size).to(device)

                # Forward
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(
                    text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len, spk_ids)

                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

                # print loss
                if (current_step+1)%10==0:
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(total_loss, mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss)
                    print(str2 + '\n')

                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)
                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()

                current_step +=1
    ###########################################
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, waveglow, melgan, text, sentence, prefix=''):
    sentence = sentence[:150] # long filename will result in OS Error
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    
    # create dir
    if not os.path.exists(os.path.join(hp.test_path, hp.dataset)):
        os.makedirs(os.path.join(hp.test_path, hp.dataset))    
    
    # generate wav
    if hp.use_spk_embed:
        hp.batch_size = 3
        # select speakers
        # TODO
        spk_ids = torch.tensor(list(inv_spk_table.keys())[5:5+hp.batch_size]).to(torch.int64).to(device)
        text = text.repeat(hp.batch_size, 1)
        src_len = src_len.repeat(hp.batch_size)
        mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len, speaker_ids=spk_ids)
        
        mel_mask = get_mask_from_lengths(mel_len, None)
        mel_mask = mel_mask.unsqueeze(-1).expand(mel_postnet.size())
        silence = (torch.ones(mel_postnet.size()) * -5).to(device)
        mel = torch.where(~mel_mask, mel, silence)
        mel_postnet = torch.where(~mel_mask, mel_postnet, silence)
        
        mel_torch = mel.transpose(1, 2).detach()
        mel_postnet_torch = mel_postnet.transpose(1, 2).detach()

        if waveglow is not None:
            wavs = utils.waveglow_infer_batch(mel_postnet_torch, waveglow)
        if melgan is not None:
            wavs = utils.melgan_infer_batch(mel_postnet_torch, melgan)        
            
        for i, spk_id in enumerate(spk_ids):
            spker = inv_spk_table[int(spk_id)]
            mel_postnet_i = mel_postnet[i].cpu().transpose(0, 1).detach()
            f0_i = f0_output[i].detach().cpu().numpy()
            energy_i = energy_output[i].detach().cpu().numpy()
            mel_mask_i = mel_mask[i]
            wav_i = wavs[i]
            
            # output
            base_dir_i = os.path.join(hp.test_path, hp.dataset, "step {}".format(args.step), spker)
            os.makedirs(base_dir_i, exist_ok=True)
            #use griffin-lim 
            Audio.tools.inv_mel_spec(mel_postnet_torch, os.path.join(base_dir_i,'{}_griffin_lim_{}.wav'.format(prefix, sentence)))
            path_i = os.path.join(base_dir_i, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence))
            soundfile.write(path_i, wav_i, hp.sampling_rate)
            utils.plot_data([(mel_postnet_i.numpy(), f0_i, energy_i)], 
                            ['Synthesized Spectrogram'], 
                            filename=os.path.join(base_dir_i, '{}_{}.png'.format(prefix, sentence)))
            
    else:
        spk_ids = None
        mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len, speaker_ids=spk_ids)
        mel_torch = mel.transpose(1, 2).detach()
        mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
        mel = mel[0].cpu().transpose(0, 1).detach()
        mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
        f0_output = f0_output[0].detach().cpu().numpy()
        energy_output = energy_output[0].detach().cpu().numpy()
        
        Audio.tools.inv_mel_spec(mel_postnet, os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))
        if waveglow is not None:
            utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(hp.test_path, hp.dataset, '{}_{}_{}_{}.wav'.format(prefix, hp.vocoder, spker, sentence)))
        if melgan is not None:
            utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(hp.test_path, hp.dataset, '{}_{}_{}_{}.wav'.format(prefix, hp.vocoder, spker, sentence)))
        
        utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=600000)
    parser.add_argument('--input', action="store_true", default=False)
    args = parser.parse_args()
    
    if args.input:
        sentence = input("Please enter an English sentence : ")
        sentences = [sentence]
        
    else:
        sentences = ["Weather forecast for tonight: dark.",
                     "I put a dollar in a change machine. Nothing changed.",
                     "“No comment” is a comment.",
                     "So far, this is the oldest I’ve been.",
                     "I am in shape. Round is a shape."
                ]
    dataset = Dataset('val.txt')
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)
        
    #model = get_FastSpeech2(args.step).to(device)
    model = get_FastSpeech2(args.step, loader).to(device)
    melgan = waveglow = None
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan()
        
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)
        
    print("Synthesizing...")
    for sentence in sentences:
        text = preprocess(sentence)
        synthesize(model, waveglow, melgan, text, sentence, prefix='step_{}'.format(args.step))
