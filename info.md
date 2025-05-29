# practice

TODO:
1 - find datasets
2 - update models info
3 - design experiments
4 - conduct the experiments
5 - prepare the results to the DOC
6 - update previous DOC
7 - add practice part to the DOC
8 - add diary to the DOC

## dataset

- general knowledge test
- specific knowledge test (generate using AI)

У якості попередньої теми для генерацій було обрано перелік уявних компаній, які мають певну історію, перелік продуктів, штат людей з детальним описом хто чим займається, за що відповідає, та багато додаткової інформації.

- Датасет повинен містити не менше 20000 слів.
- Тренувальний датасет повинен містити не менше 100 прикладів, у той час як тестовий датасет повинен містити не менше 50 прикладів.

## metrics

- точність відповідей, у якості основного критерія порівняння (test set accuracy, test set answer relevance)
- швидкодія конкретної конфігурації моделі (latency)
- ціна обробки конкретної конфігурації (cost)
- ґрунтовність моделі на додаткових знаннях (groundedness)
- відповідність отриманого контексту до початкових запитань (context relevance)

## models

old:
Gemini Pro 1.5 та Llama 3.2 11B

<!-- gpt-4o-mini
gemini-2.0-flash-lite
? -->

## options

- pretraining (mask prediction/next token prediction) [?]
- fine-tuning (partial/LoRA/QLoRA)
- prompting (zero-shot/few-shot)
- distillation (пропонується порівняти якість прямого дотреновування меншої моделі на тренувальному датасеті та дистиляції знань з більшої моделі в меншу)
- RAG (with/without agent | Neo4j)

## notes

- structure text into structural information
- TruLens(?)
