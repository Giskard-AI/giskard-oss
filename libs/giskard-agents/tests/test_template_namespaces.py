import tempfile
from pathlib import Path

from giskard.agents.templates.prompts_manager import PromptsManager


async def test_default_namespace_template():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prompts_manager = PromptsManager()
        prompts_manager.set_default_prompts_path(tmp_dir)

        template_path = Path(tmp_dir) / "hello.j2"
        template_path.write_text("Hello, {{ name }}!")

        messages = await prompts_manager.render_template(
            "hello.j2", {"name": "Orlande de Lassus"}
        )

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello, Orlande de Lassus!"


async def test_namespaced_template():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prompts_manager = PromptsManager()
        prompts_manager.add_prompts_path(tmp_dir, "test")
        prompts_manager.add_prompts_path(
            Path(__file__).parent / "data" / "prompts", "test2"
        )

        template_path = Path(tmp_dir) / "hello.j2"
        template_path.write_text("Hello, {{ name }}!")

        messages = await prompts_manager.render_template(
            "test::hello.j2", {"name": "Orlande de Lassus"}
        )

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello, Orlande de Lassus!"
