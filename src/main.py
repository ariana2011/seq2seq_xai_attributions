from data import Data
from  inseq_output import InseqModel

if __name__ == "__main__":
    # data = Data()
    # train_data, validation_data = data.hugging_face_wmt("wmt19", "de-en")
    # sample_sentences = data.range_data()
    
    inseq_model = InseqModel()
    inseq_out = inseq_model.inseq_process("This is a sample of input sentence", x_ai="input_x_gradient" )
    print(inseq_out)
    
    # print(sample_sentences)