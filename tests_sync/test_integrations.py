import json

from sync.integrations import remove_legacy_pi_package


def test_remove_legacy_pi_package_preserves_unrelated_settings(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "packages": [
                    "../../.funes/integrations/pi",
                    {"source": "../../.funes/integrations/pi", "autoload": True},
                    "github:user/keep-me",
                ],
                "theme": "dark",
            }
        ),
        encoding="utf-8",
    )

    assert remove_legacy_pi_package(tmp_path) is True
    assert json.loads(settings.read_text(encoding="utf-8")) == {
        "packages": ["github:user/keep-me"],
        "theme": "dark",
    }


def test_remove_legacy_pi_package_is_idempotent_and_preserves_invalid_json(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{"packages":["github:user/keep-me"]}\n', encoding="utf-8")
    assert remove_legacy_pi_package(tmp_path) is False
    original = settings.read_bytes()
    settings.write_text("not-json\n", encoding="utf-8")
    assert remove_legacy_pi_package(tmp_path) is False
    assert settings.read_bytes() == b"not-json\n"
    assert original != settings.read_bytes()


def test_remove_legacy_pi_package_honors_pi_lock(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{"packages":["../../.funes/integrations/pi"]}\n', encoding="utf-8")
    lock = settings.with_name("settings.json.lock")
    lock.mkdir()

    assert remove_legacy_pi_package(tmp_path) is False
    assert json.loads(settings.read_text(encoding="utf-8"))["packages"] == [
        "../../.funes/integrations/pi"
    ]


def test_remove_legacy_pi_package_preserves_settings_symlink(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    target = tmp_path / "dotfiles" / "pi-settings.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        '{"packages":["../../.funes/integrations/pi","github:user/keep-me"]}\n',
        encoding="utf-8",
    )
    settings.parent.mkdir(parents=True)
    settings.symlink_to(target)

    assert remove_legacy_pi_package(tmp_path) is True
    assert settings.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8"))["packages"] == [
        "github:user/keep-me"
    ]


def test_remove_legacy_pi_package_does_not_follow_package_aliases(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    legacy = tmp_path / ".funes" / "integrations" / "pi"
    actual = tmp_path / "packages" / "different-source"
    actual.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(actual, target_is_directory=True)
    settings.write_text(
        json.dumps({"packages": ["../../.funes/integrations/pi", str(actual)]}),
        encoding="utf-8",
    )

    assert remove_legacy_pi_package(tmp_path) is True
    assert json.loads(settings.read_text(encoding="utf-8"))["packages"] == [
        str(actual)
    ]


def test_remove_legacy_pi_package_ignores_non_object_json(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("[]\n", encoding="utf-8")
    assert remove_legacy_pi_package(tmp_path) is False
    assert settings.read_text(encoding="utf-8") == "[]\n"
