"""Locally constructed BPE tokenizer; never fetches remote models."""
from tokenizers import Tokenizer, AddedToken, models, pre_tokenizers, decoders
import yaml


def pairs(raw):
    lines = raw.decode().splitlines()
    if len(lines)%2: raise ValueError('Malformed token/id pairs')
    return {lines[i]:int(lines[i+1]) for i in range(0,len(lines),2)}


def tokenizer(pack):
    vocab=pairs(pack.read('asr/token/vocab.txt'))
    merges=[tuple(line.split()) for line in pack.read('asr/token/merges.txt').decode().splitlines() if line]
    tok=Tokenizer(models.BPE(vocab,merges,unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel()
    special=pairs(pack.read('asr/token/add_special_tokens.txt'))
    tok.add_special_tokens([AddedToken(s,special=True,normalized=False) for s in special])
    return tok


def prompt_parts(pack):
    config=yaml.safe_load(pack.read('config.yaml'))
    templates=config['objects']['prompt_processor']['prompt_template_map']
    prompt=next(t['prompt_template'] for t in templates if t['name']=='default')
    for key in ['BOT_NAME','BOT_DESC','USER_MODIFY','DIALOG','HOTWORDS','CUR_CONTEXT']:
        prompt=prompt.replace('@@'+key+'@@','')
    return prompt.split('@@AUDIO@@')


def normalize_output(text):
    """Apply model-emitted ITN annotations; preserve malformed/unknown objects."""
    import json,re
    decoder=json.JSONDecoder();result=[];i=0
    while i<len(text):
        if text[i]=='{':
            try:
                obj,end=decoder.raw_decode(text[i:])
                if isinstance(obj,dict) and isinstance(obj.get('itn'),str) and isinstance(obj.get('asr'),str):
                    result.append(obj['itn']);i+=end;continue
            except (ValueError,TypeError):pass
        result.append(text[i]);i+=1
    return re.sub(r'〈[^〈〉]*\|A〉','', ''.join(result))
