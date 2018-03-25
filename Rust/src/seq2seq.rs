extern crate tensorflow;

use tensorflow::Tensor;

pub struct Seq2seq {
	encoder_cell: Tensor,
	decoder_cell: Tensor,
	vocab_size: u32,
	inp_emb_size: u32,
}

