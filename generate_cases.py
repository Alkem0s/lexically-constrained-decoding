import json

def make_tc(src, dir_, req, forb, rew=None, pen=None):
    return {
        "source": src,
        "direction": dir_,
        "forbidden_words": forb,
        "required_words": req,
        "penalty_words": pen if pen is not None else forb,
        "reward_words": rew if rew is not None else req
    }

hpo_en_tr = [
    # single easy
    make_tc("The cat sat on the mat.", "EN→TR", ["halı"], ["kedi"]),
    make_tc("She bought a new car yesterday.", "EN→TR", ["araba"], ["yeni"]),
    make_tc("The sun rises in the east.", "EN→TR", ["güneş"], ["doğu"]),
    make_tc("I love eating fresh fruit.", "EN→TR", ["meyve"], ["taze"]),
    make_tc("He is a good student.", "EN→TR", ["öğrenci"], ["iyi"]),
    # single hard
    make_tc("The doctor advised the patient to exercise.", "EN→TR", ["hekim"], ["doktor"]),
    make_tc("Climate change is a serious problem.", "EN→TR", ["iklim değişikliği"], []),
    make_tc("The algorithm optimizes the loss function.", "EN→TR", ["algoritma"], ["fonksiyon"]),
    make_tc("The lawyer presented the evidence in court.", "EN→TR", ["kanıt"], ["avukat"]),
    make_tc("Nuclear physics requires advanced mathematics.", "EN→TR", ["nükleer"], ["fizik"]),
    # multiple easy
    make_tc("The dog chased the cat.", "EN→TR", ["köpek", "kedi"], ["kovaladı"]),
    make_tc("He drank coffee in the morning.", "EN→TR", ["kahve", "sabah"], ["içti"]),
    make_tc("They walked through the green park.", "EN→TR", ["yeşil", "park"], ["yürüdü"]),
    make_tc("The boy kicked the red ball.", "EN→TR", ["kırmızı", "top"], ["çocuk"]),
    make_tc("She read a book in the library.", "EN→TR", ["kitap", "kütüphane"], ["okudu"]),
    # multiple hard
    make_tc("Artificial intelligence is transforming the global economy.", "EN→TR", ["yapay zeka", "dönüşüm"], ["ekonomi"]),
    make_tc("The software engineer fixed the bug.", "EN→TR", ["yazılım mühendisi", "çözmek"], ["hata"]),
    make_tc("Global warming threatens polar bear habitats.", "EN→TR", ["küresel ısınma", "tehdit"], ["kutup"]),
    make_tc("The stock exchange fluctuates daily.", "EN→TR", ["borsa", "dalgalanmak"], ["günlük"]),
    make_tc("Cybersecurity experts detected a massive breach.", "EN→TR", ["siber güvenlik", "ihlal"], ["büyük"]),
    # exclusion only
    make_tc("The fast red car won the race.", "EN→TR", [], ["hızlı", "kırmızı"]),
    make_tc("The weather is extremely cold today.", "EN→TR", [], ["hava", "çok"]),
    make_tc("He spoke very loudly during the meeting.", "EN→TR", [], ["çok", "toplantı"]),
    make_tc("The beautiful painting was sold at auction.", "EN→TR", [], ["güzel", "tablo"])
]

hpo_tr_en = [
    # single easy
    make_tc("Kedi halının üzerinde oturdu.", "TR→EN", ["feline"], ["cat"]),
    make_tc("O dün yeni bir araba aldı.", "TR→EN", ["vehicle"], ["car"]),
    make_tc("Güneş doğudan doğar.", "TR→EN", ["sun"], ["east"]),
    make_tc("Taze meyve yemeyi severim.", "TR→EN", ["fruit"], ["fresh"]),
    make_tc("O iyi bir öğrenci.", "TR→EN", ["student"], ["good"]),
    # single hard
    make_tc("Doktor hastaya ilaç yazdı.", "TR→EN", ["physician"], ["doctor"]),
    make_tc("Yapay zeka iş dünyasını değiştiriyor.", "TR→EN", ["transforming"], ["change"]),
    make_tc("Algoritma kayıp fonksiyonunu optimize eder.", "TR→EN", ["algorithm"], ["function"]),
    make_tc("Avukat kanıtı mahkemeye sundu.", "TR→EN", ["evidence"], ["lawyer"]),
    make_tc("Nükleer fizik ileri matematik gerektirir.", "TR→EN", ["nuclear"], ["physics"]),
    # multiple easy
    make_tc("Büyük köpek küçük kediyi kovaladı.", "TR→EN", ["hound", "feline"], ["dog", "cat"]),
    make_tc("Sabahları kahve içmeyi severim.", "TR→EN", ["morning", "beverage"], ["coffee"]),
    make_tc("Yeşil parkın içinden yürüdüler.", "TR→EN", ["green", "park"], ["walked"]),
    make_tc("Çocuk kırmızı topa vurdu.", "TR→EN", ["red", "ball"], ["boy"]),
    make_tc("Kütüphanede bir kitap okudu.", "TR→EN", ["book", "library"], ["read"]),
    # multiple hard
    make_tc("Modern mimari estetik ve işlevselliği birleştirir.", "TR→EN", ["contemporary", "aesthetics"], ["modern"]),
    make_tc("Enflasyon oranları geçen yıla göre düştü.", "TR→EN", ["inflation", "decrease"], ["down"]),
    make_tc("Küresel ısınma kutup ayılarının yaşam alanlarını tehdit ediyor.", "TR→EN", ["global warming", "threatens"], ["polar"]),
    make_tc("Borsa günlük olarak dalgalanıyor.", "TR→EN", ["stock market", "fluctuates"], ["daily"]),
    make_tc("Siber güvenlik uzmanları büyük bir ihlal tespit etti.", "TR→EN", ["cybersecurity", "breach"], ["massive"]),
    # exclusion only
    make_tc("Hızlı kırmızı araba yarışı kazandı.", "TR→EN", [], ["fast", "red"]),
    make_tc("Hava bugün aşırı soğuk.", "TR→EN", [], ["weather", "very"]),
    make_tc("Toplantı sırasında çok yüksek sesle konuştu.", "TR→EN", [], ["very", "meeting"]),
    make_tc("Güzel tablo açık artırmada satıldı.", "TR→EN", [], ["beautiful", "painting"])
]

eval_en_tr = [
    # single easy
    make_tc("The bird flew over the tall tree.", "EN→TR", ["kuş"], ["uçtu"]),
    make_tc("I drink water every morning.", "EN→TR", ["su"], ["içerim"]),
    make_tc("He plays football with his friends.", "EN→TR", ["futbol"], ["arkadaş"]),
    make_tc("She likes to watch movies.", "EN→TR", ["film"], ["izlemek"]),
    make_tc("The train arrived late.", "EN→TR", ["tren"], ["geç"]),
    make_tc("The book is on the table.", "EN→TR", ["kitap"], ["masa"]),
    # single hard
    make_tc("The patient needs immediate surgery.", "EN→TR", ["operasyon"], ["ameliyat"]),
    make_tc("The CEO resigned yesterday.", "EN→TR", ["istifa"], ["ayrıldı"]),
    make_tc("Quantum computing is the future.", "EN→TR", ["kuantum"], ["gelecek"]),
    make_tc("The stock market crashed.", "EN→TR", ["borsa"], ["piyasa"]),
    make_tc("The renewable energy sector is growing.", "EN→TR", ["yenilenebilir"], ["enerji"]),
    make_tc("The detective found a crucial clue.", "EN→TR", ["ipucu"], ["buldu"]),
    # multiple easy
    make_tc("A small dog barked at the big cat.", "EN→TR", ["küçük", "kedi"], ["köpek"]),
    make_tc("She wrote a letter to her mother.", "EN→TR", ["mektup", "anne"], ["yazdı"]),
    make_tc("The strong wind blew away the tent.", "EN→TR", ["rüzgar", "çadır"], ["güçlü"]),
    make_tc("He opened the door and walked inside.", "EN→TR", ["kapı", "içeri"], ["açtı"]),
    make_tc("They ate dinner at a fancy restaurant.", "EN→TR", ["akşam", "restoran"], ["yemek"]),
    make_tc("The blue sky was full of clouds.", "EN→TR", ["mavi", "bulut"], ["gökyüzü"]),
    # multiple hard
    make_tc("The international space station orbits the Earth.", "EN→TR", ["uzay istasyonu", "yörünge"], ["dünya"]),
    make_tc("Machine translation uses neural networks.", "EN→TR", ["makine çevirisi", "sinir ağları"], ["çeviri"]),
    make_tc("The central bank raised interest rates.", "EN→TR", ["merkez bankası", "faiz"], ["oran"]),
    make_tc("The archaeological dig revealed ancient artifacts.", "EN→TR", ["arkeolojik", "eserler"], ["kazı"]),
    make_tc("The new legislation will affect tax brackets.", "EN→TR", ["mevzuat", "vergi"], ["yeni"]),
    make_tc("Sustainable agriculture relies on water conservation.", "EN→TR", ["sürdürülebilir", "koruma"], ["tarım"]),
    # exclusion only
    make_tc("The quick brown fox jumps over the lazy dog.", "EN→TR", [], ["hızlı", "kahverengi"]),
    make_tc("We must protect the environment.", "EN→TR", [], ["çevre"]),
    make_tc("The concert was very loud.", "EN→TR", [], ["konser", "çok"]),
    make_tc("She is studying for her final exams.", "EN→TR", [], ["çalışıyor"]),
    make_tc("The city is known for its beautiful architecture.", "EN→TR", [], ["güzel"]),
    make_tc("The museum was closed on Monday.", "EN→TR", [], ["kapalı"])
]

eval_tr_en = [
    # single easy
    make_tc("Kuş yüksek ağacın üzerinden uçtu.", "TR→EN", ["avian"], ["bird"]),
    make_tc("Her sabah su içerim.", "TR→EN", ["consume"], ["drink"]),
    make_tc("Arkadaşlarıyla futbol oynar.", "TR→EN", ["soccer"], ["football"]),
    make_tc("Film izlemeyi sever.", "TR→EN", ["cinema"], ["movies"]),
    make_tc("Tren geç geldi.", "TR→EN", ["delayed"], ["late"]),
    make_tc("Kitap masanın üzerinde.", "TR→EN", ["novel"], ["book"]),
    # single hard
    make_tc("Hasta acil ameliyat olmalı.", "TR→EN", ["surgery"], ["operation"]),
    make_tc("CEO dün istifa etti.", "TR→EN", ["resigned"], ["quit"]),
    make_tc("Kuantum bilgisayarlar gelecektir.", "TR→EN", ["computing"], ["computers"]),
    make_tc("Borsa çöktü.", "TR→EN", ["crashed"], ["fell"]),
    make_tc("Yenilenebilir enerji sektörü büyüyor.", "TR→EN", ["renewable"], ["green"]),
    make_tc("Dedektif önemli bir ipucu buldu.", "TR→EN", ["evidence"], ["clue"]),
    # multiple easy
    make_tc("Küçük köpek büyük kediye havladı.", "TR→EN", ["tiny", "feline"], ["small", "cat"]),
    make_tc("Annesine bir mektup yazdı.", "TR→EN", ["epistle", "mother"], ["letter"]),
    make_tc("Güçlü rüzgar çadırı uçurdu.", "TR→EN", ["gale", "tent"], ["wind"]),
    make_tc("Kapıyı açtı ve içeri girdi.", "TR→EN", ["doorway", "inside"], ["door"]),
    make_tc("Lüks bir restoranda akşam yemeği yediler.", "TR→EN", ["dinner", "establishment"], ["restaurant"]),
    make_tc("Mavi gökyüzü bulutlarla doluydu.", "TR→EN", ["azure", "clouds"], ["blue", "sky"]),
    # multiple hard
    make_tc("Uluslararası uzay istasyonu dünyanın yörüngesinde dönüyor.", "TR→EN", ["orbital", "station"], ["orbit"]),
    make_tc("Makine çevirisi sinir ağlarını kullanır.", "TR→EN", ["machine translation", "neural networks"], ["translation"]),
    make_tc("Merkez bankası faiz oranlarını artırdı.", "TR→EN", ["central bank", "interest"], ["rates"]),
    make_tc("Arkeolojik kazı antik eserleri ortaya çıkardı.", "TR→EN", ["excavation", "artifacts"], ["dig"]),
    make_tc("Yeni mevzuat vergi dilimlerini etkileyecek.", "TR→EN", ["legislation", "tax"], ["law"]),
    make_tc("Sürdürülebilir tarım suyun korunmasına dayanır.", "TR→EN", ["sustainable", "conservation"], ["agriculture"]),
    # exclusion only
    make_tc("Hızlı kahverengi tilki tembel köpeğin üzerinden atlar.", "TR→EN", [], ["quick", "brown"]),
    make_tc("Çevreyi korumalıyız.", "TR→EN", [], ["protect"]),
    make_tc("Konser çok gürültülüydü.", "TR→EN", [], ["concert", "very"]),
    make_tc("Final sınavları için çalışıyor.", "TR→EN", [], ["studying"]),
    make_tc("Şehir güzel mimarisiyle bilinir.", "TR→EN", [], ["beautiful"]),
    make_tc("Müze pazartesi günü kapalıydı.", "TR→EN", [], ["closed"])
]

hpo_data = {"EN_TR": hpo_en_tr, "TR_EN": hpo_tr_en}
eval_data = {"EN_TR": eval_en_tr, "TR_EN": eval_tr_en}

with open("test_cases_hpo.json", "w", encoding="utf-8") as f:
    json.dump(hpo_data, f, ensure_ascii=False, indent=2)

with open("test_cases_eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print(f"Generated test_cases_hpo.json ({len(hpo_en_tr)} EN-TR, {len(hpo_tr_en)} TR-EN)")
print(f"Generated test_cases_eval.json ({len(eval_en_tr)} EN-TR, {len(eval_tr_en)} TR-EN)")
