from src.topic_prase.doc_parse_new import DocumentAnalyze
DA = DocumentAnalyze()
topic_id = int(input('话题索引'))
DA.build(topic_id=topic_id)
id = int(input('id'))
connection_level = input('content or topic?')


DA.run(connection_level =connection_level,id=id)
