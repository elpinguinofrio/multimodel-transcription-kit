"""transcription.toml: providers (accounts) by type, and models best-first with their providers.

    [[aistudio]]                       # one block per account; types: aistudio, antigravity, openai
    name = "kerocat"                   # yours; models refer to it, receipts record it
    key_env = "GEMINI_API_KEY"         # env variable holding its key (optional for openai)
    base_url = "http://127.0.0.1:8300" # optional; default: the type's public address
    costs_money = true                 # optional: tried only after every free route failed
    parallel_chunks = 4                # optional: parts of a long recording sent at once (default 1)

    [[model]]                          # best first: the order of these blocks is the ranking
    name = "gemini-3.8-flash"
    providers = ["agy-main:gemini-3.8-flash-low", "kerocat"]   # "provider:id" if it names it differently
    word_times = true                  # optional: may supply word times (same order)

    [[profile]]                        # optional: a named model order a caller picks with --profile
    name = "balanced"
    models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]   # only these, in this order

    [defaults]
    profile = "quality"                # the profile used when the caller names none

The rest ([secrets], [defaults], [limits], [vocabulary], [usage], [fusion]) is as in the INI.

``load`` turns the file into the ConfigParser shape the CLI already uses: one
``[model.<model>@<provider>]`` section per route, ``[models] ranking`` with every free
route in model order and then every paid route, ``timing_ranking`` likewise for
word_times models, and ``[model_routes]`` mapping each model name to its routes.
Each ``[[profile]]`` becomes ``[profiles] <name>`` (its models' routes, free then paid);
``[defaults] profile`` also sets ``[models] ranking`` to that profile's order.
``convert_ini`` writes a TOML file from an existing INI config.
"""

import configparser
import json

TYPES = {"aistudio": "gemini", "antigravity": "agy", "openai": "openai"}
DEFAULT_BASE_URL = {"openai": "https://api.openai.com/v1"}
PROVIDER_KEYS = {"name", "key_env", "base_url", "costs_money", "parallel_chunks", "note"}
MODEL_KEYS = {"name", "providers", "word_times", "prompt", "temperature", "chunk_seconds", "merge_model",
              "capabilities", "rpm", "tpm", "rpd"}
PROFILE_KEYS = {"name", "models", "note"}
PLAIN_SECTIONS = ("secrets", "defaults", "limits", "vocabulary", "usage", "fusion")
LIST_JOIN = {("vocabulary", "terms"): ", ", ("vocabulary", "corrections"): "\n", ("limits", "tiers"): ", "}


class ConfigError(ValueError):
    """The TOML config is not valid; the message names the block and the setting."""


def _text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(_text(v) for v in value)
    return str(value)


def _read(path):
    try:
        import tomllib
    except ImportError:                                   # Python 3.10
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise ConfigError("reading a .toml config needs Python 3.11+ (or the tomli package)") from exc
    with open(path, "rb") as handle:
        try:
            return tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc


def _providers(data):
    providers = {}
    for type_name in TYPES:
        blocks = data.get(type_name, [])
        if not isinstance(blocks, list):
            raise ConfigError(f"[{type_name}] must be written [[{type_name}]] (one block per account)")
        for block in blocks:
            name = block.get("name")
            if not name or not isinstance(name, str):
                raise ConfigError(f"a [[{type_name}]] block has no name")
            unknown = set(block) - PROVIDER_KEYS
            if unknown:
                raise ConfigError(f"[[{type_name}]] {name}: unknown setting(s) {', '.join(sorted(unknown))}")
            if name in providers:
                raise ConfigError(f"two providers are named {name!r}")
            if type_name == "antigravity" and (block.get("base_url") or block.get("key_env")):
                raise ConfigError(f"[[antigravity]] {name}: the agy CLI takes no base_url or key_env")
            parallel = block.get("parallel_chunks", 1)
            if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
                raise ConfigError(f"[[{type_name}]] {name}: parallel_chunks must be a whole number of at "
                                  f"least 1, not {parallel!r}")
            providers[name] = {"type": type_name, **block}
    return providers


def _routes(data, providers):
    """[(model block, route name, section options)] in block order."""
    models = data.get("model", [])
    if not isinstance(models, list) or not models:
        raise ConfigError("add at least one [[model]] block")
    seen, routes = set(), []
    for block in models:
        name = block.get("name")
        if not name or not isinstance(name, str):
            raise ConfigError("a [[model]] block has no name")
        if name in seen:
            raise ConfigError(f"two [[model]] blocks are named {name!r}")
        seen.add(name)
        unknown = set(block) - MODEL_KEYS
        if unknown:
            raise ConfigError(f"[[model]] {name}: unknown setting(s) {', '.join(sorted(unknown))}")
        refs = block.get("providers")
        if not refs or not isinstance(refs, list):
            raise ConfigError(f"[[model]] {name}: list its providers, e.g. providers = [\"my-account\"]")
        passthrough = {k: _text(v) for k, v in block.items() if k not in ("name", "providers", "word_times")}
        for ref in refs:
            provider_name, _, model_id = str(ref).partition(":")
            provider = providers.get(provider_name)
            if provider is None:
                raise ConfigError(f"[[model]] {name}: no provider named {provider_name!r} "
                                  f"(known: {', '.join(providers) or 'none'})")
            model_id = model_id or name
            kind = TYPES[provider["type"]]
            base_url = provider.get("base_url") or DEFAULT_BASE_URL.get(provider["type"])
            options = {"kind": kind, "model": model_id, "account": provider_name,
                       "limit_key": f"{provider_name}/{model_id}", **passthrough}
            if base_url:
                options["base_url"] = base_url
            if provider.get("key_env"):
                options["requires_env"] = provider["key_env"]
            if provider.get("costs_money"):
                options["costs_money"] = "true"
            if provider.get("parallel_chunks", 1) != 1:
                options["parallel_chunks"] = str(provider["parallel_chunks"])
            routes.append((block, f"{name}@{provider_name}", options))
    return routes


def load(path):
    """A ConfigParser with the CLI's sections, built from a transcription.toml."""
    data = _read(path)
    known = set(PLAIN_SECTIONS) | set(TYPES) | {"model", "profile"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown section(s) {', '.join(sorted(unknown))}; "
                          f"known: {', '.join(sorted(known))}")
    parser = configparser.ConfigParser(interpolation=None)
    for section in PLAIN_SECTIONS:
        values = data.get(section)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ConfigError(f"[{section}] must be a table of settings")
        parser.add_section(section)
        for key, value in values.items():
            joiner = LIST_JOIN.get((section, key))
            parser[section][key] = joiner.join(map(str, value)) if joiner and isinstance(value, list) else _text(value)
    routes = _routes(data, _providers(data))
    for _, route, options in routes:
        parser[f"model.{route}"] = options
    free = [r for _, r, o in routes if o.get("costs_money") != "true"]
    paid = [r for _, r, o in routes if o.get("costs_money") == "true"]
    timed = [(b, r, o) for b, r, o in routes if b.get("word_times")]
    parser["models"] = {
        "ranking": ", ".join(free + paid),
        "timing_ranking": ", ".join([r for _, r, o in timed if o.get("costs_money") != "true"]
                                    + [r for _, r, o in timed if o.get("costs_money") == "true"])}
    parser["model_routes"] = {}
    for block, route, _ in routes:
        existing = parser["model_routes"].get(block["name"])
        parser["model_routes"][block["name"]] = f"{existing}, {route}" if existing else route
    profiles = _profiles(data, routes)
    if profiles:
        parser["profiles"] = profiles
    default = parser.get("defaults", "profile", fallback="").strip()
    if default:
        if default not in profiles:
            raise ConfigError(f"[defaults] profile: no profile named {default!r} "
                              f"(known: {', '.join(profiles) or 'none'})")
        parser["models"]["ranking"] = profiles[default]
    return parser


def _profiles(data, routes):
    """{profile name: its routes as a ranking}: listed models in order, free routes before paid."""
    blocks = data.get("profile", [])
    if not isinstance(blocks, list):
        raise ConfigError("[profile] must be written [[profile]] (one block per profile)")
    models = {}
    for block, route, options in routes:
        models.setdefault(block["name"], []).append((route, options.get("costs_money") == "true"))
    profiles = {}
    for block in blocks:
        name = block.get("name")
        if not name or not isinstance(name, str):
            raise ConfigError("a [[profile]] block has no name")
        unknown = set(block) - PROFILE_KEYS
        if unknown:
            raise ConfigError(f"[[profile]] {name}: unknown setting(s) {', '.join(sorted(unknown))}")
        if name in profiles:
            raise ConfigError(f"two profiles are named {name!r}")
        listed = block.get("models")
        if not listed or not isinstance(listed, list):
            raise ConfigError(f"[[profile]] {name}: list its models, e.g. models = [\"my-model\"]")
        missing = [m for m in listed if m not in models]
        if missing:
            raise ConfigError(f"[[profile]] {name}: no model named {missing[0]!r} (known: {', '.join(models)})")
        chosen = [r for m in listed for r in models[m]]
        profiles[name] = ", ".join([r for r, paid in chosen if not paid] + [r for r, paid in chosen if paid])
    return profiles


# ───────────────────────── INI -> TOML ──────────────────────────

def _q(value):
    return json.dumps(value, ensure_ascii=False)


def _value(section, key, raw):
    if (section, key) == ("vocabulary", "terms"):
        from .vocabulary import parse_terms
        return "[" + ", ".join(_q(t) for t in parse_terms(raw)) + "]"
    if (section, key) == ("vocabulary", "corrections"):
        from .vocabulary import parse_corrections
        items = [f"{h} -> {r}" for h, r in parse_corrections(raw)]
        return "[\n" + "".join(f"  {_q(i)},\n" for i in items) + "]"
    if (section, key) == ("limits", "tiers"):
        return "[" + ", ".join(_q(t.strip()) for t in raw.split(",") if t.strip()) + "]"
    lowered = raw.strip().lower()
    if lowered in ("true", "false", "yes", "no", "on", "off"):
        return "true" if lowered in ("true", "yes", "on") else "false"
    try:
        return str(int(raw)) if raw.strip().lstrip("-").isdigit() else str(float(raw))
    except ValueError:
        return _q(raw)


def convert_ini(parser):
    """TOML text equivalent to an INI config (models from [models] ranking, then timing-only ones)."""
    def names(key):
        return [n.strip() for n in parser.get("models", key, fallback="").split(",")
                if n.strip() and parser.has_section(f"model.{n.strip()}")]
    ranking, timing = names("ranking"), names("timing_ranking")
    # One list holds both orders: a timing-only model goes just before the next model of the
    # old timing order, so word times keep their order; text order changes only by that insertion.
    order = list(ranking)
    for index, name in enumerate(timing):
        if name in order:
            continue
        later = next((n for n in timing[index + 1:] if n in order), None)
        order.insert(order.index(later) if later else len(order), name)
    out, notes = [], []
    for section in PLAIN_SECTIONS:
        if parser.has_section(section) and parser[section]:
            out.append(f"[{section}]")
            out += [f"{key} = {_value(section, key, raw)}" for key, raw in parser[section].items()]
            out.append("")
    providers, refs = {}, {}
    for name in order:
        options = parser[f"model.{name}"]
        kind = options.get("kind", "")
        if kind == "gemini":
            key_env = options.get("requires_env") or "GEMINI_API_KEY"
            pname = "aistudio" if key_env == "GEMINI_API_KEY" else f"aistudio-{key_env.lower()}"
            providers.setdefault(pname, ("aistudio", {"key_env": key_env}))
        elif kind == "groq":
            key_env = options.get("requires_env") or "GROQ_API_KEY"
            pname = "groq" if key_env == "GROQ_API_KEY" else f"groq-{key_env.lower()}"
            providers.setdefault(pname, ("openai", {"key_env": key_env,
                                                    "base_url": "https://api.groq.com/openai/v1"}))
        elif kind == "http":
            base = options.get("endpoint", "").split("/audio/")[0]
            pname = next((p for p, (_, o) in providers.items() if o.get("base_url") == base), None)
            if pname is None:
                pname = "local" if "local" not in providers else f"local-{len(providers)}"
                providers[pname] = ("openai", {"base_url": base})
        else:
            raise ConfigError(f"[model.{name}]: kind {kind!r} cannot be converted")
        if options.get("parallel_chunks"):
            settings = providers[pname][1]
            parallel = int(options["parallel_chunks"])
            if settings.setdefault("parallel_chunks", parallel) != parallel:
                raise ConfigError(f"[model.{name}]: parallel_chunks {parallel} differs from another model "
                                  f"on the same account ({settings['parallel_chunks']}); set one value")
        model_id = options.get("model") or name
        refs[name] = pname if model_id == name else f"{pname}:{model_id}"
    out.append("# Providers: one block per account. type = the block name: aistudio, antigravity, openai.")
    for pname, (ptype, settings) in providers.items():
        out.append(f"[[{ptype}]]")
        out.append(f"name = {_q(pname)}")
        out += [f"{key} = {_q(value)}" for key, value in settings.items()]
        out.append("")
    timing_only = [n for n in timing if n not in ranking]
    if timing_only:
        notes.append(f"# NOTE: {', '.join(timing_only)} supplied only word times before; every model block is "
                     f"also a text fallback now, placed to keep the word-time order")
    timed_in_order = [n for n in order if n in timing]
    if timed_in_order != timing:
        notes.append(f"# NOTE: timing_ranking was {', '.join(timing)}; word times now follow the model order: "
                     f"{', '.join(timed_in_order)}")
    out.append("# Models, best first: the order of these blocks is the ranking.")
    out += notes
    for name in order:
        options = parser[f"model.{name}"]
        out.append("[[model]]")
        out.append(f"name = {_q(name)}")
        out.append(f"providers = [{_q(refs[name])}]")
        if name in timing:
            out.append("word_times = true")
        for key, raw in options.items():
            if key in ("kind", "model", "endpoint", "requires_env", "parallel_chunks"):
                continue
            if key not in MODEL_KEYS:
                raise ConfigError(f"[model.{name}]: setting {key!r} has no TOML equivalent")
            out.append(f"{key} = {_value('model', key, raw)}")
        out.append("")
    if parser.has_section("profiles"):
        out.append("# Profiles: named model orders; a caller picks one with --profile.")
        for name, raw in parser["profiles"].items():
            listed = [m.strip() for m in raw.split(",") if m.strip()]
            missing = [m for m in listed if m not in order]
            if missing:
                raise ConfigError(f"[profiles] {name}: no model named {missing[0]!r}")
            out += ["[[profile]]", f"name = {_q(name)}", "models = [" + ", ".join(map(_q, listed)) + "]", ""]
    return "\n".join(out).rstrip() + "\n"
