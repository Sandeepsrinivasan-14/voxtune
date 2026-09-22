from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.db import Project, get_session

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    task: str = Field(pattern="^(stt|tts|llm)$")


def _out(p: Project) -> dict:
    return {"id": p.id, "name": p.name, "task": p.task, "created_at": p.created_at.isoformat()}


@router.post("", status_code=201)
def create_project(body: ProjectIn, s: Session = Depends(get_session)):
    p = Project(name=body.name, task=body.task)
    s.add(p)
    s.commit()
    return _out(p)


@router.get("")
def list_projects(s: Session = Depends(get_session)):
    return [_out(p) for p in s.query(Project).order_by(Project.id.desc()).all()]


@router.get("/{project_id}")
def get_project(project_id: int, s: Session = Depends(get_session)):
    p = s.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return _out(p)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: int, s: Session = Depends(get_session)):
    p = s.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    s.delete(p)
    s.commit()
