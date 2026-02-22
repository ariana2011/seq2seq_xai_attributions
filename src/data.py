#%%
import datasets
from transformers import MarianMTModel, MarianTokenizers
class Data():
    def hugging_face_wmt(self, data_name = "wmt19", language = "de-en"):
        dataset = datasets.load_dataset(data_name, language)
        train_data = dataset["train"]
        validation_data = dataset['validation']
        return train_data, validation_data
    
    def range_data(self, range_num = 5):
        """returaning a few sents"""
        train_data, validation_data = self.hugging_face_wmt("wmt19", "de-en")
        sample_size = range_num
        sample_sentences = [example['translation']['en'] for example in train_data.select(range(sample_size))]
        return sample_sentences
    
if __name__ == "__main__":
    data = Data()
    train_data, validation_data = data.hugging_face_wmt("wmt19", "de-en")
    sample_sentences = data.range_data()

    print("Train Data:")
    print(train_data)

    print("\nSample English Sentences:")
    for i, sentence in enumerate(sample_sentences, 1):
        print(f"{i}: {sentence}")

