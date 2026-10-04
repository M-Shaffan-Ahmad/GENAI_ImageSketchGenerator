"""Update local MLflow file artifact URIs after cloning/unpacking elsewhere."""
import sqlite3
from pathlib import Path

root=Path(__file__).resolve().parents[1]
database=root/'experiments/tracking.db'
if not database.exists():raise SystemExit('No prepared experiments/tracking.db found')
with sqlite3.connect(database) as db:
    for identifier,uri in db.execute('SELECT experiment_id,artifact_location FROM experiments').fetchall():
        if '/tracking_artifacts/' in uri:
            relative=uri.split('/tracking_artifacts/',1)[1]
            db.execute('UPDATE experiments SET artifact_location=? WHERE experiment_id=?',((root/'experiments/tracking_artifacts'/relative).as_uri(),identifier))
    for identifier,uri in db.execute('SELECT run_uuid,artifact_uri FROM runs').fetchall():
        if '/tracking_artifacts/' in uri:
            relative=uri.split('/tracking_artifacts/',1)[1]
            db.execute('UPDATE runs SET artifact_uri=? WHERE run_uuid=?',((root/'experiments/tracking_artifacts'/relative).as_uri(),identifier))
print('Tracking artifact paths now point to this checkout.')
