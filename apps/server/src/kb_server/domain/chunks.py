import uuid

from salemdex_retrieval import Dense, Many, One, Schema


class Guarded(Schema):
    access_group_ids = Many(uuid.UUID, indexed=True)
    collection_id = One(uuid.UUID, indexed=True)
    document_id = One(uuid.UUID, indexed=True)
    document_version = One(int)


class KbChunks(Guarded):
    # id: uuid5(document_id, f"{version}:{position}")
    position = One(int)
    headings = Many(str)
    dense = Dense(size=1024, distance="cosine")
