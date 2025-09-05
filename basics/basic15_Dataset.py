# wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
# wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
"""
	! Huggingface dataset kütüphanesinin kullanımı
"""

"""
! ÖRNEK VERİSETİ

	"data": [{
		"paragraphs": [{
			"qas": [{
				"question": "Quando è iniziata la crisi petrolifera del 1973?",
				"answers": [
					{
						"text": "ottobre 1973",
						"answer_start": 43
					},
					{
						"text": "ottobre 1973",
						"answer_start": 43
					},
					{
						"text": "ottobre 1973",
						"answer_start": 43
					},
					{
						"text": "ottobre",
						"answer_start": 43
					},
					{
						"text": "1973",
						"answer_start": 25
					}
				],
				"id": "5725b33f6a3fe71400b8952d"
			},
			"context": "La crisi petrolifera del 1973 iniziò nell' ottobre 1973 quando i membri dell' Organizzazione dei Paesi esportatori di petrolio arabo (OAPEC, composta dai membri arabi dell' OPEC più Egitto e Siria) proclamarono un embargo petrolifero. Alla fine dell' embargo, nel marzo 1974, il prezzo del petrolio era salito da 3 dollari al barile a quasi 12 dollari a livello mondiale; i prezzi americani erano notevolmente più elevati. L' embargo ha causato una crisi petrolifera, o \"shock\", con molti effetti a breve e lungo termine sulla politica globale e sull' economia globale. Più tardi fu chiamato il \"primo shock petrolifero\", seguito dalla crisi petrolifera del 1979, definita il \"secondo shock petrolifero\"."
		}],
		"title": "Crisi energetica (1973)"
    }]
"""


from datasets import load_dataset
from rich import print


data_files = {
    "train": "temp/dataset/SQuAD/SQuAD_it-train.json",
    "test": "temp/dataset/SQuAD/SQuAD_it-test.json"
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

print(squad_it_dataset)






