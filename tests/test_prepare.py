import json
from pathlib import Path

import pandas as pd
import pytest

from resume_classifier.ml.dataset import load_and_split
from resume_classifier.ml.prepare import (
    build_text,
    connected_groups,
    parse_nested,
    prepare_cv_export,
    professional_spheres,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", []),
        ("[]", []),
        ("not a list", []),
        ("42", []),
        ('[{"a":1}]', [{"a": 1}]),
        ("[{'a': True}]", [{"a": True}]),
        ('{"a":1}', [{"a": 1}]),
    ],
)
def test_nested_parser_handles_json_and_python_literals_without_eval(
    value: str,
    expected: list[dict[str, object]],
) -> None:
    assert parse_nested(value) == expected


def test_target_is_distinct_professional_sphere_not_profession_or_position() -> None:
    value = (
        "[{'codeProfessionalSphere': 'Sales', 'codeProfession': '123'}, "
        "{'codeProfessionalSphere': 'Sales'}]"
    )
    assert professional_spheres(value) == ["Sales"]
    assert professional_spheres('[{"codeProfession":"123"}]') == []


def test_feature_extraction_excludes_target_identifiers_and_company_names() -> None:
    record = {
        "positionName": "Developer",
        "professionList": "TARGET_SENTINEL",
        "candidateId": "PRIVATE_ID",
        "gender": "GENDER_SENTINEL",
        "workExperienceList": "[{'jobTitle':'Engineer', 'demands':'Build API', "
        "'companyName':'COMPANY_SENTINEL'}]",
        "educationList": "[{'qualification':'Computing', 'instituteName':'SCHOOL_SENTINEL'}]",
        "hardSkills": "['Python', 'SQL']",
    }
    text = build_text(record)
    assert "Developer" in text
    assert "Build API" in text
    assert "Python" in text
    for forbidden in [
        "TARGET_SENTINEL",
        "PRIVATE_ID",
        "GENDER_SENTINEL",
        "COMPANY_SENTINEL",
        "SCHOOL_SENTINEL",
    ]:
        assert forbidden not in text
    assert len(build_text({"positionName": "x" * 30000})) == 20000


def test_groups_connect_same_candidate_and_duplicate_text_transitively() -> None:
    groups = connected_groups(["a", "a", "b", "c"], ["first", " second ", "SECOND", "unrelated"])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def raw_frame() -> pd.DataFrame:
    rows = []
    for category in ["Sales", "InformationTechnology"]:
        for index in range(30):
            rows.append(
                {
                    "id": f"{category}-{index}",
                    "candidateId": f"{category}-{index // 2}",
                    "positionName": f"{category} resume sample {index}",
                    "professionList": json.dumps([{"codeProfessionalSphere": category}]),
                    "workExperienceList": "[]",
                    "hardSkills": "['SQL']",
                }
            )
    return pd.DataFrame(rows)


def test_cv_export_has_group_split_and_complete_lineage(tmp_path: Path) -> None:
    path = tmp_path / "cv.csv"
    raw_frame().to_csv(path, index=False)
    splits = load_and_split(path)
    groups = [set(part.group_id) for part in [splits.train, splits.validation, splits.test]]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert sum(len(part) for part in [splits.train, splits.validation, splits.test]) == 60
    assert splits.preparation["raw_rows"] == 60
    assert splits.preparation["prepared_rows"] == 60


def test_preparation_reports_dropped_rows_and_rare_classes() -> None:
    raw = raw_frame()
    rare = raw.iloc[:1].copy()
    rare["id"] = "rare"
    rare["positionName"] = "rare unique"
    rare["professionList"] = '[{"codeProfessionalSphere":"Rare"}]'
    missing = rare.copy()
    missing["professionList"] = "[]"
    raw = pd.concat([raw, raw.iloc[:1], rare, missing], ignore_index=True)
    prepared, report = prepare_cv_export(raw)
    assert len(prepared) == 60
    assert report["discarded_rows"]["duplicate_text_and_label"] == 1
    assert report["discarded_rows"]["missing_or_multiple_target"] == 1
    assert report["unsupported_classes"] == {"Rare": 1}


def test_preparation_rejects_wrong_schema_or_empty_export() -> None:
    with pytest.raises(ValueError, match="must have"):
        prepare_cv_export(pd.DataFrame({"x": ["x"]}))
    frame = raw_frame()
    frame["professionList"] = "[]"
    with pytest.raises(ValueError, match="No labeled"):
        prepare_cv_export(frame)
