
VOICEMAIL_PHRASES = [
    'voice mail', 'voicemail', 'message', 'leave voicemail', 'grabar un mensaje', 'mensaje', 'marcando', 'mensaje predeterminado',
    'contestador', 'deje un mensaje', 'dejar un mensaje', 'mensaje de voz', 'despues del tono', 'oir el tono', 'movistar', 'predeterminado',
    'grabe su mensaje', 'buzon', 'tono', 'finalizar', 'grabe', 'buzon de voz', 'no se encuentra', 'no esta disponible', 'no contesta', 'correo',
    'servicio', 'este es'
]

def is_text_voicemail(text: str):
    print('is_text_voicemail  Verifying if transcripot has voicemail phrases')
    content = text.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace(" ", "")
    print('is_text_voicemail normalized transcript:', content)
    for phrase in VOICEMAIL_PHRASES:
        phrase = phrase.replace(" ", "")
        if phrase in content:
            print('is_text_voicemail Phrase found:', phrase)
            return True
    print('is_text_voicemail No Phrases found')
    return False





def _test_voicemail():

    text = 'leave a message'
    is_voicemail = is_text_voicemail(text)
    print(is_voicemail)

if __name__ == '__main__':
    _test_voicemail()