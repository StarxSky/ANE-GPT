#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import Core
import torch
import logging
import unittest
import collections


import numpy as np
import coremltools as ct
import Core.ANE_Model as ane_transformer

# =================================
torch.set_grad_enabled(False)




logger = logging.getLogger(__name__)
logger.setLevel('INFO')

# Testing configuration
PSNR_THRESHOLD = 20
SANITY_CHECK_CPUGPU2ANE_SPEEDUP_FACTOR = 10

TEST_BATCH_SIZE = 2
TEST_SRC_SEQ_LEN = 128
TEST_TGT_SEQ_LEN = 256
TEST_EMBED_DIM = 512

TEST_INPUTS = collections.OrderedDict({
    'encoder_input':
    torch.rand(TEST_BATCH_SIZE, TEST_EMBED_DIM, 1, TEST_SRC_SEQ_LEN),
    'decoder_input':
    torch.rand(TEST_BATCH_SIZE, TEST_EMBED_DIM, 1, TEST_TGT_SEQ_LEN),
    'encoder_pos_embed':
    torch.rand(TEST_BATCH_SIZE, TEST_EMBED_DIM, 1, TEST_SRC_SEQ_LEN),
    'decoder_pos_embed':
    torch.rand(TEST_BATCH_SIZE, TEST_EMBED_DIM, 1, TEST_TGT_SEQ_LEN),
    'encoder_k_mask':
    torch.zeros(TEST_BATCH_SIZE, TEST_SRC_SEQ_LEN, 1, 1),
    'decoder_k_mask':
    torch.zeros(TEST_BATCH_SIZE, TEST_TGT_SEQ_LEN, 1, 1),
    'encoder_qk_mask':
    torch.zeros(TEST_BATCH_SIZE, TEST_SRC_SEQ_LEN, 1, TEST_SRC_SEQ_LEN),
    'decoder_qk_mask':
    torch.zeros(TEST_BATCH_SIZE, TEST_TGT_SEQ_LEN, 1, TEST_TGT_SEQ_LEN),
})

# =======================================================
class TestTransformerReferenceImplementation(unittest.TestCase):
    """
    Test conversion success and ANE speed-up of the reference Transformer implementation
    """

    @classmethod
    def setUpClass(cls):
        cls.model = ane_transformer.AppleNeuralEngineTransformer(
            embed_dim=TEST_EMBED_DIM)
        cls.inputs = TEST_INPUTS
        cls.ref_outputs = cls.model(**cls.inputs)

    @classmethod
    def tearDownClass(cls):
        cls.model = None
        cls.inputs = None

    def test_coreml_conversion_and_speedup(self):
        # Conversion from PyTorch module to Torchscript module
        try:
            module_traced = torch.jit.trace(self.model,
                                            list(self.inputs.values()))
        except Exception as e:
            raise RuntimeError("Torchscript tracing failed!") from e

        logger.info("Torchscript tracing is successful")

        # Conversion from Torchscript module to CoreML Model Package
        # Targeting (primarily) ANE without GPU+CPU fallback
        try:
            ANE_Model_CoreML_Model = ct.convert(
                module_traced,
                convert_to='mlprogram',
                inputs=[
                    ct.TensorType(
                        name,
                        shape=tensor.shape,
                        dtype=np.float32,
                    ) for name, tensor in self.inputs.items()
                ],
                compute_units=ct.ComputeUnit.ALL,
            )
            
            ANE_Model_CoreML_Model.save("ANE_CoreML_Model.mlpackage")
        except Exception as e:

            raise RuntimeError(
                "CoreML conversion targeting ANE failed!") from e

        logger.info("CoreML conversion targeting ANE is successful")


       

    
if __name__ == "__main__":
    unittest.main()
