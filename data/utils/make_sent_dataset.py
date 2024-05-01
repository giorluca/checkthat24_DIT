import re
import json

def text_to_sentence_dataset(data):
    # List to store dictionaries representing sentences with metadata
    sentence_data = []
    for sample in data:
        # Tokenize the text into sentences
        text = sample['text'].replace('\n', '\n ')
        sentences = re.findall(r'[^.!?]*[.!?]', text)

        for sentence in sentences:
            # Calculate the start and end indices of the sentence
            start_index = text.index(sentence)
            end_index = start_index + len(sentence)

            # Adjust spans to sentence-level
            adjusted_sentence_spans = []
            for anno in sample['annotations']:
                span_start = anno['start']
                span_end = anno['end']
                if span_start >= start_index and span_end <= end_index:
                    adjusted_start = span_start - start_index
                    adjusted_end = span_end - start_index + 1
                    monitor = sentence[adjusted_start:adjusted_end+1]
                    adjusted_sentence_spans.append({'start':adjusted_start, 'end':adjusted_end, 'tag':anno['tag']})

            #If there are no annotations for the sentence
            if not adjusted_sentence_spans:
                sentence_dict = {
                    'text': sentence,
                    'article_id': sample['article_id'],
                    'lang': sample['lang'],
                    'annotations': [],
                    'label':0
                }
            else:
                # Create a dictionary representing the sentence with metadata
                sentence_dict = {
                    'text': sentence,
                    'article_id': sample['article_id'],
                    'lang': sample['lang'],
                    'annotations': adjusted_sentence_spans,
                    'label':1
                }

            sentence_data.append(sentence_dict)

    return sentence_data

def main():
    data = json.load(open('/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold/train_gold.json', 'r', encoding='utf8'))
    sentence_data = text_to_sentence_dataset(data)

    with open("/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold/train_gold_sentences_new.json", "w", encoding='utf-8') as outfile: 
        json.dump(sentence_data, outfile, ensure_ascii = False)

if __name__ == "__main__":
    main()
