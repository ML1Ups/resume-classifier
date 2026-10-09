import ast
import json
import re
from collections import Counter
from hashlib import sha256

import pandas as pd

TEXT_FIELDS = (
    "positionName",
    "hardSkills",
    "softSkills",
    "skills",
    "workExperienceList",
    "educationList",
    "additionalEducationList",
    "education",
    "additionalInformation",
)
NESTED_KEYS = {
    "workExperienceList": {"jobTitle", "demands", "achievements"},
    "educationList": {"qualification", "speciality", "faculty"},
    "additionalEducationList": {"qualification", "speciality", "faculty", "courseName", "name"},
}


def parse_nested(value: str) -> list:
    if not value or value.strip() in {"[]", "{}", "null", "None"}:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if isinstance(parsed, list):
        return parsed
    return [parsed] if isinstance(parsed, dict) else []


def professional_spheres(value: str) -> list[str]:
    return sorted(
        {
            str(row["codeProfessionalSphere"]).strip()
            for row in parse_nested(value)
            if isinstance(row, dict) and row.get("codeProfessionalSphere")
        }
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def build_text(record: dict) -> str:
    pieces = []
    for field in TEXT_FIELDS:
        value = record.get(field, "")
        if not value:
            continue
        if field in NESTED_KEYS:
            for item in parse_nested(value):
                if isinstance(item, dict):
                    pieces.extend(
                        str(item[key]) for key in sorted(NESTED_KEYS[field]) if item.get(key)
                    )
        elif field in {"hardSkills", "softSkills", "skills"}:
            pieces.extend(item for item in parse_nested(value) if isinstance(item, str))
        else:
            pieces.append(value)
    text = re.sub(r"<[^>]*>", " ", ". ".join(pieces))
    return re.sub(r"\s+", " ", text).strip()[:20000]


def connected_groups(candidates: list[str], texts: list[str]) -> list[str]:
    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, (candidate, text) in enumerate(zip(candidates, texts, strict=True)):
        for key in [("person", candidate or f"missing-{index}"), ("text", normalize_text(text))]:
            if key in owners:
                root, previous = find(index), find(owners[key])
                parent[max(root, previous)] = min(root, previous)
            else:
                owners[key] = index
    return [sha256(f"group-{find(i)}".encode()).hexdigest()[:16] for i in range(len(texts))]


def prepare_cv_export(raw: pd.DataFrame, min_class_rows: int = 10) -> tuple[pd.DataFrame, dict]:
    required = {"id", "candidateId", "professionList", "positionName"}
    if not required.issubset(raw.columns):
        raise ValueError("CV export must have id, candidateId, professionList and positionName")
    records = []
    discarded = Counter()
    input_class_counts = Counter()
    for row in raw.to_dict(orient="records"):
        spheres = professional_spheres(row["professionList"])
        if len(spheres) != 1:
            discarded["missing_or_multiple_target"] += 1
            continue
        label = spheres[0]
        input_class_counts[label] += 1
        text = build_text(row)
        if not text:
            discarded["empty_text"] += 1
            continue
        records.append(
            {
                "record_id": sha256(row["id"].encode()).hexdigest()[:16],
                "candidate": row["candidateId"],
                "text": text,
                "category": label,
                "_has_position": bool(row.get("positionName", "").strip()),
                "_has_work": bool(parse_nested(row.get("workExperienceList", ""))),
                "_has_hard_skills": bool(parse_nested(row.get("hardSkills", ""))),
                "_has_soft_skills": bool(parse_nested(row.get("softSkills", ""))),
                "_has_education": bool(parse_nested(row.get("educationList", ""))),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No labeled non-empty resumes in CV export")
    frame["group_id"] = connected_groups(frame["candidate"].tolist(), frame["text"].tolist())
    frame["normalized"] = frame["text"].map(normalize_text)
    duplicate_mask = frame.duplicated(["normalized", "category"])
    discarded["duplicate_text_and_label"] = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].copy()
    counts = frame["category"].value_counts()
    group_counts = frame.groupby("category")["group_id"].nunique()
    supported = counts.index[
        (counts >= min_class_rows) & (group_counts.reindex(counts.index) >= 10)
    ]
    unsupported = frame.loc[~frame["category"].isin(supported), "category"].value_counts().to_dict()
    discarded["insufficient_class_support"] = int(sum(unsupported.values()))
    frame = frame.loc[frame["category"].isin(supported)].drop(columns=["candidate", "normalized"])
    if frame["category"].nunique() < 2:
        raise ValueError("Need at least two supported professional spheres")
    report = {
        "format": "cv_target structured export",
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "prepared_rows": len(frame),
        "target": "professionList[*].codeProfessionalSphere (one distinct sphere per resume)",
        "text_fields": list(TEXT_FIELDS),
        "max_text_characters": 20000,
        "excluded_from_features": [
            "professionList",
            "typicalPosition",
            "id",
            "candidateId",
            "idUser",
            "innerInfo",
            "birthday",
            "gender",
            "age",
            "companyName",
            "instituteName",
            "contact fields",
        ],
        "discarded_rows": dict(discarded),
        "unsupported_classes": unsupported,
        "raw_class_counts": dict(input_class_counts),
        "prepared_class_counts": frame["category"].value_counts().to_dict(),
        "minimum_rows_and_groups_per_class": min_class_rows,
        "independent_groups": int(frame["group_id"].nunique()),
        "largest_group_rows": int(frame["group_id"].value_counts().max()),
        "group_policy": "connected components of same candidateId OR same normalized text",
        "dedup_policy": "same normalized text and target: keep first, after group construction",
    }
    return frame.reset_index(drop=True), report
