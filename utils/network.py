import logging
import re

def get_encoder_and_decoder_params(model):
    """Filter model parameters into two groups: encoder and decoder."""
    logger = logging.getLogger(__name__)
    enc_params = []
    dec_params = []
    for k, v in model.named_parameters():
        if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
            # print(k)
            enc_params.append(v)
            logger.info(" Enc. parameter: {}".format(k))
        else:
            # print(k)
            dec_params.append(v)
            logger.info(" Dec. parameter: {}".format(k))
    return enc_params, dec_params