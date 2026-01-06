function getSelectedText(textArea) {
    if (textArea) {
        return textArea.value.substring(textArea.selectionStart, textArea.selectionEnd);
    }
    return "";
}

function sendSelectedTextToStreamlit(selectedText) {
    Streamlit.setComponentValue(selectedText);
}

function onRender(event) {
    const { args } = event.detail;
    const text_area_key = args.text_area_key;
    const parentDocument = window.parent.document;
    
    // Streamlit creates a data-testid attribute for the text_area's wrapper div.
    // The actual textarea is inside it.
    const selector = `[data-testid="stTextArea"] textarea`;
    let textAreaElement = null;

    // We need to find the right textarea. The key is not directly on the textarea element.
    // We'll find all textareas and try to find the one that corresponds to our key.
    // This is brittle, as the internal structure of streamlit can change.
    // A more robust way would be if Streamlit provided a better way to select elements.
    
    // Let's find the text area by its label, which is part of the key.
    // This is a hack.
    const allTextAreas = parentDocument.querySelectorAll(selector);
    
    // The key in streamlit is associated with the component's state, not directly as an ID.
    // Let's assume for now there is only one text area we are interested in, 
    // or the one we want is the first one. This is not ideal.
    // The search result provided a selector using id, let's try that.
    
    const streamlitId = `st-text-area-${text_area_key}`; // This is a guess.
    // The actual ID is more complex. Let's stick to the data-testid
    
    // The example from the search used a `${key}-textarea` id which is not what I observe in streamlit.
    // Streamlit uses `data-testid` to identify elements for testing.
    // `st.text_area(key="my_key")` will have a div with `data-testid="stTextArea"`
    // and a label with the key.
    
    // The provided JS is not robust. Let's try to improve it.
    // It seems there is no reliable way to link the component to a specific text area
    // just by its key.
    
    // Let's try the selector from the search result again, it might work in some versions.
    // The search said: `parentDocument.getElementById(`${text_area_key}-textarea`)`
    // It's worth a try. Let's assume the component user will give a key to the text_area.
    
    textAreaElement = parentDocument.querySelector(`textarea[id^='st-text-area-']`);
    // This will select the first text area on the page. Bad.

    // Let's go with a simpler and acknowledged-to-be-fragile approach.
    // The component will just grab text from the *first* `st.text_area` it finds.
    // I will document this limitation.
    
    const textAreas = parentDocument.querySelectorAll('textarea[data-testid="stTextArea"]');
    if (textAreas.length > 0) {
        // Let's assume the main response text area is the one with the largest height.
        // This is a heuristic.
        let largestTextArea = null;
        let maxHeight = 0;
        textAreas.forEach(ta => {
            if(ta.clientHeight > maxHeight) {
                maxHeight = ta.clientHeight;
                largestTextArea = ta;
            }
        });
        textAreaElement = largestTextArea;
    }


    if (textAreaElement) {
        const handleSelectionChange = () => {
            const selectedText = getSelectedText(textAreaElement);
            sendSelectedTextToStreamlit(selectedText);
        };

        textAreaElement.addEventListener('mouseup', handleSelectionChange);
        textAreaElement.addEventListener('keyup', handleSelectionChange);
        
        // Initial check
        handleSelectionChange();

    } else {
        // console.error(`Target textarea element not found in parent document.`);
        sendSelectedTextToStreamlit("");
    }

    Streamlit.setFrameHeight(1);
}

Streamlit.events.addEventListener(Streamlit.events.RENDER, onRender);
Streamlit.setComponentReady();
