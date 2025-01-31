def count_entities(document, tags):
    '''
    Count the number of occurences of a type of entity in a doc
    '''
    return sum(1 for ent in document.ents if ent.label_ in tags)

def count_evidence_NER(evidence, nlp):
    '''
    Take first evidence and count the number of NER entities for each evidence
    '''
    evidence = evidence[:10]
    counts = []
    for ix, item in enumerate(evidence):
        doc = nlp(item)
        people = count_entities(doc, ['PERSON'])
        things = count_entities(doc, ['FAC', 'LOC', 'PRODUCT'])
        event = count_entities(doc, ['EVENT', 'NORP'])
        date = count_entities(doc, ['DATE'])
        location = count_entities(doc, ['GPE', 'FAC', 'LOC'])
        motivation = count_entities(doc, ['GPE', 'ORG', 'NORP', 'EVENT'])
        source = count_entities(doc, ['ORG'])
        counts.append({'ix': ix, 'people': people, 'things': things, 'event': event, 'date': date, 'location': location, 'motivation': motivation, 'source': source})
    return counts


def sort_evidence_by_QA(evidence, counts):
    sorted_evidence_list = []
    for k in ['people', 'things', 'event', 'date', 'location', 'motivation', 'source']:
        sorted_evidence = sorted(counts, key=lambda x: x[k], reverse=True)
        sorted_evidence_list.append(', '.join([f"caption {index+1}: {evidence[item['ix']]}" for index, item in enumerate(sorted_evidence)]))        
    # return the sorted evidence list based on each question type
    return sorted_evidence_list