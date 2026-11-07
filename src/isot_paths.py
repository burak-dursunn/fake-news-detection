"""Drive'daki hazırlanmış ISOT dosyalarını indirme yapmadan bulur."""
from pathlib import Path

PREPARED_NAMES = ("isot_temiz.csv", "isot_split_seed42.csv", "isot_hazirlama_raporu.json")


def find_prepared_isot(directories: list[Path]) -> dict[str, Path] | None:
    """İlk tam klasörü seçer; eksik kopyaları sessizce ham veriyle değiştirmez.

    Dizin sırası açık önceliktir. Ayrı klasörlerdeki parçalar birleştirilmez.
    """
    partial = []
    checked = set()
    for directory in directories:
        directory = Path(directory)
        if directory in checked:
            continue
        checked.add(directory)
        if not directory.is_dir():
            print(f"Hazır ISOT klasörü yok: {directory}", flush=True)
            continue
        entries = {}
        for path in directory.iterdir():
            if path.is_file() and path.name.casefold() in PREPARED_NAMES:
                key = path.name.casefold()
                if key in entries:
                    raise ValueError(f"Aynı isimde birden çok ISOT dosyası var: {directory}")
                entries[key] = path
        missing = set(PREPARED_NAMES) - entries.keys()
        if not missing:
            print(f"Hazır ISOT bulundu: {directory}", flush=True)
            return entries
        print(f"{directory}: eksik dosyalar: {', '.join(sorted(missing))}", flush=True)
        if entries:
            partial.append((directory, sorted(missing)))
    if partial:
        detail = "; ".join(f"{path}: {', '.join(missing)}" for path, missing in partial)
        raise FileNotFoundError(f"Hazır ISOT kopyası eksik; ham veri indirilmedi. Eksikler: {detail}")
    return None
