from divine_semantics.src.db_helper import get_db_connection, cantica_id2name
from divine_semantics.src.experiment import load_model
import config


conn = get_db_connection()
cur = conn.cursor()

cantica_id = 2

print(cantica_id2name(cantica_id))