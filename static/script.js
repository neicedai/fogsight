document.addEventListener('DOMContentLoaded', () => {
    const config = {
        apiBaseUrl: '', 
        defaultLang: 'zh',
    };

    const translations = {
        heroTitle: { zh: "在此赋予概念以生命，转瞬之间", en: "Bring Concepts to Life Here" },
        startCreatingTitle: { zh: "开始创作", en: "Start Creating" },
        githubrepo: { zh: "Github 开源仓库", en: "Fogsight Github Repo" },
        officialWebsite: { zh: "通向 AGI 之路社区", en: "WaytoAGI Open Source Community" },
        groupChat: { zh: "联系我们/加入交流群", en: "Contact Us" },
        placeholders: {
            zh: ["微积分的几何原理", "冒泡排序","热寂", "黑洞是如何形成的"],
            en: ["What is Heat Death?", "How are black holes formed?", "What is Bubble Sort?"]
        },
        newChat: { zh: "新对话", en: "New Chat" },
        newChatTitle: { zh: "新对话", en: "New Chat" },
        chatPlaceholder: {
            zh: "AI 生成结果具有随机性，您可在此输入修改意见",
            en: "Results are random. Enter your modifications here for adjustments."
        },
        sendTitle: { zh: "发送", en: "Send" },
        agentThinking: { zh: "Fogsight Agent 正在进行思考与规划，请稍后。这可能需要数十秒至数分钟...", en: "Fogsight Agent is thinking and planning, please wait..." },
        generatingCode: { zh: "生成代码中...", en: "Generating code..." },
        codeComplete: { zh: "代码已完成", en: "Code generated" },
        openInNewWindow: { zh: "在新窗口中打开", en: "Open in new window" },
        saveAsHTML: { zh: "保存为 HTML", en: "Save as HTML" },
        exportAsVideo: { zh: "导出为视频", en: "Export as Video" },
        generateVoiceover: { zh: "生成配音", en: "Generate Voiceover" },
        downloadVoiceover: { zh: "下载配音", en: "Download Voiceover" },
        featureComingSoon: { zh: "该功能正在开发中，将在不久的将来推出。\n 请关注我们的官方 GitHub 仓库以获取最新动态！", en: "This feature is under development and will be available soon.\n Follow our official GitHub repository for the latest updates!" },
        visitGitHub: { zh: "访问 GitHub", en: "Visit GitHub" },
        errorMessage: { zh: "抱歉，服务出现了一点问题。请稍后重试。", en: "Sorry, something went wrong. Please try again later." },
        errorFetchFailed: {zh: "LLM服务不可用，请稍后再试", en: "LLM service is unavailable. Please try again later."},
        errorTooManyRequests: {zh: "今天已经使用太多，请明天再试", en: "Too many requests today. Please try again tomorrow."},
        errorLLMParseError: {zh: "返回的动画代码解析失败，请调整提示词重新生成。", en: "Failed to parse the returned animation code. Please adjust your prompt and try again."},
        voiceoverPlaceholder: { zh: "生成的配音将在这里显示", en: "Generated voiceover will appear here." },
        voiceoverAutoGenerating: { zh: "自动配音生成中...", en: "Generating Chinese voiceover..." },
        voiceoverAutoReady: { zh: "自动配音已生成", en: "Chinese voiceover ready" },
        voiceoverAutoFailed: { zh: "自动配音失败", en: "Auto voiceover failed." },
        voiceoverAutoSubtitleMissing: { zh: "未识别到中文字幕，无法生成配音。", en: "No Chinese subtitles detected. Unable to create voiceover." },
        voiceoverModalTitle: { zh: "生成动画配音", en: "Generate Animation Voiceover" },
        voiceoverModalDescription: { zh: "输入旁白文本并上传说话人参考音频，系统会调用 TTS 服务生成配音。", en: "Provide narration text and a speaker reference audio to generate a voiceover via the TTS service." },
        voiceoverTextLabel: { zh: "旁白文本", en: "Narration text" },
        voiceoverVoiceLabel: { zh: "说话人参考音频", en: "Speaker reference audio" },
        voiceoverEmotionLabel: { zh: "情感参考音频（可选）", en: "Emotion reference audio (optional)" },
        voiceoverGenerateButton: { zh: "开始生成", en: "Generate" },
        voiceoverGeneratingStatus: { zh: "配音生成中...", en: "Generating voiceover..." },
        voiceoverSuccessStatus: { zh: "配音已生成", en: "Voiceover ready" },
        voiceoverErrorStatus: { zh: "配音生成失败，请重试。", en: "Voiceover generation failed. Please try again." },
        voiceoverMissingAudio: { zh: "请先生成配音后再下载。", en: "Generate a voiceover before downloading." },
        voiceoverServiceUnavailable: { zh: "配音服务暂不可用，请稍后再试。", en: "Voiceover service is unavailable. Please try again later." },
        voiceoverSpeakerRequired: { zh: "请上传说话人参考音频。", en: "Please upload a speaker reference audio file." },
        voiceoverTextRequired: { zh: "请输入要朗读的文本。", en: "Please enter narration text." },
    };

    const CHINESE_CHAR_REGEX = /[\u3400-\u9fff]/;

    let currentLang = config.defaultLang;
    const body = document.body;
    const initialForm = document.getElementById('initial-form');
    const initialInput = document.getElementById('initial-input');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatLog = document.getElementById('chat-log');
    const newChatButton = document.getElementById('new-chat-button');
    const languageSwitcher = document.getElementById('language-switcher');
    const placeholderContainer = document.getElementById('animated-placeholder');
    const featureModal = document.getElementById('feature-modal');
    const modalGitHubButton = document.getElementById('modal-github-button');
    const modalCloseButton = document.getElementById('modal-close-button');
    const voiceoverModal = document.getElementById('voiceover-modal');
    const voiceoverCloseButton = document.getElementById('voiceover-close-button');
    const voiceoverForm = document.getElementById('voiceover-form');
    const voiceoverTextInput = document.getElementById('voiceover-text');
    const voiceoverSpeakerInput = document.getElementById('voiceover-speaker');
    const voiceoverEmotionInput = document.getElementById('voiceover-emo');
    const voiceoverStatus = document.getElementById('voiceover-status');
    const voiceoverSubmitButton = document.getElementById('voiceover-submit-button');

    const templates = {
        user: document.getElementById('user-message-template'),
        status: document.getElementById('agent-status-template'),
        code: document.getElementById('agent-code-template'),
        player: document.getElementById('animation-player-template'),
        error: document.getElementById('agent-error-template'),
    };

    class LLMParseError extends Error {
        constructor(message, code = 'LLM_UNKNOWN_ERROR') {
            super(message);
            this.name = 'LLMParseError';
            this.code = code;
        }
    }

    let conversationHistory = [];
    let accumulatedCode = '';
    let placeholderInterval;
    let activeVoiceoverPlayer = null;
    let activeVoiceoverTopic = '';

    function resetVoiceoverStatus() {
        if (!voiceoverStatus) return;
        voiceoverStatus.textContent = '';
        voiceoverStatus.className = 'voiceover-status';
    }

    function setVoiceoverButtonTranslation(button, key) {
        if (!button) return;
        const label = button.querySelector('span[data-translate-key]');
        if (!label) return;
        if (key) label.dataset.translateKey = key;
        const translation = translations[key]?.[currentLang];
        if (translation) label.textContent = translation;
    }

    function updateVoiceoverStatus(playerElement, translationKey, stateClass = '', customText = null) {
        if (!playerElement) return;
        const container = playerElement.querySelector('.voiceover-container');
        if (!container) return;
        container.classList.remove('empty', 'loading', 'error', 'success');
        if (stateClass) container.classList.add(stateClass);
        const message = customText ?? translations[translationKey]?.[currentLang] ?? '';
        const placeholder = document.createElement('p');
        placeholder.className = 'voiceover-placeholder';
        if (translationKey) {
            placeholder.dataset.translateKey = translationKey;
        }
        placeholder.textContent = message;
        container.innerHTML = '';
        container.appendChild(placeholder);
    }

    function extractChineseNarrationFromHtml(htmlContent) {
        if (!htmlContent) return '';
        const parser = new DOMParser();
        const doc = parser.parseFromString(htmlContent, 'text/html');
        if (!doc) return '';

        const selectors = [
            '[data-subtitle]',
            '[data-caption]',
            '.subtitle',
            '.subtitles',
            '.captions',
            '.caption',
            '[class*="subtitle"]',
            '[class*="caption"]',
            '[id*="subtitle"]',
            '[id*="caption"]',
        ];

        const seen = new Set();
        const segments = [];

        const collectText = (rawText) => {
            if (!rawText) return;
            const normalized = rawText.replace(/\s+/g, ' ').trim();
            if (!normalized) return;
            if (!CHINESE_CHAR_REGEX.test(normalized)) return;
            const chineseParts = normalized.match(/[\u3400-\u9fff0-9\u3000-\u303F\uFF00-\uFFEF，。、！？：；“”‘’（）·—…\s]+/g);
            const chineseText = chineseParts ? chineseParts.join('').replace(/\s+/g, ' ').trim() : '';
            if (!chineseText || !CHINESE_CHAR_REGEX.test(chineseText)) return;
            if (seen.has(chineseText)) return;
            seen.add(chineseText);
            segments.push(chineseText);
        };

        const candidateElements = new Set();
        selectors.forEach((selector) => {
            doc.querySelectorAll(selector).forEach((el) => candidateElements.add(el));
        });

        candidateElements.forEach((el) => collectText(el.textContent || ''));

        if (segments.length === 0) {
            const walker = doc.createTreeWalker(doc.body || doc, NodeFilter.SHOW_TEXT, null);
            let node;
            while ((node = walker.nextNode())) {
                const parentTag = node.parentElement?.tagName?.toLowerCase();
                if (parentTag && ['script', 'style'].includes(parentTag)) continue;
                collectText(node.textContent || '');
            }
        }

        const combined = segments.join('\n');
        return combined.length > 4000 ? combined.slice(0, 4000) : combined;
    }

    async function autoGenerateVoiceoverForPlayer(playerElement, htmlContent, topic) {
        if (!playerElement) return;
        const voiceoverButton = playerElement.querySelector('.generate-voiceover');
        setVoiceoverButtonTranslation(voiceoverButton, 'voiceoverAutoGenerating');
        if (voiceoverButton) {
            voiceoverButton.disabled = true;
            voiceoverButton.classList.add('disabled');
        }

        updateVoiceoverStatus(playerElement, 'voiceoverAutoGenerating', 'loading');

        const narrationText = extractChineseNarrationFromHtml(htmlContent);
        if (!narrationText) {
            updateVoiceoverStatus(playerElement, 'voiceoverAutoSubtitleMissing', 'error');
            setVoiceoverButtonTranslation(voiceoverButton, 'voiceoverAutoFailed');
            return;
        }

        try {
            const response = await fetch(`${config.apiBaseUrl}/voiceover/auto`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: narrationText, topic: topic || '' }),
            });

            if (!response.ok) {
                const errorBody = await response.text();
                let errorMessage = translations.voiceoverAutoFailed[currentLang] || '';
                try {
                    const parsed = JSON.parse(errorBody);
                    if (parsed?.detail) errorMessage = parsed.detail;
                } catch (parseError) {
                    if (errorBody) errorMessage = errorBody;
                }
                updateVoiceoverStatus(playerElement, null, 'error', errorMessage);
                setVoiceoverButtonTranslation(voiceoverButton, 'voiceoverAutoFailed');
                showWarning(errorMessage);
                return;
            }

            const blob = await response.blob();
            attachVoiceoverToPlayer(playerElement, blob, topic || narrationText);
            setVoiceoverButtonTranslation(voiceoverButton, 'voiceoverAutoReady');
        } catch (error) {
            console.error('Auto voiceover generation failed:', error);
            const fallbackMessage = translations.voiceoverServiceUnavailable[currentLang]
                || translations.voiceoverAutoFailed[currentLang]
                || 'Voiceover service is unavailable.';
            updateVoiceoverStatus(playerElement, null, 'error', fallbackMessage);
            setVoiceoverButtonTranslation(voiceoverButton, 'voiceoverAutoFailed');
            showWarning(fallbackMessage);
        }
    }

    function openVoiceoverModal(playerElement, topic) {
        if (!voiceoverModal || !voiceoverTextInput || !voiceoverSpeakerInput) return;
        activeVoiceoverPlayer = playerElement;
        activeVoiceoverTopic = topic || '';
        voiceoverTextInput.value = topic || '';
        voiceoverSpeakerInput.value = '';
        if (voiceoverEmotionInput) voiceoverEmotionInput.value = '';
        resetVoiceoverStatus();
        voiceoverModal.classList.add('visible');
    }

    function closeVoiceoverModal() {
        if (!voiceoverModal) return;
        if (voiceoverSpeakerInput) voiceoverSpeakerInput.value = '';
        if (voiceoverEmotionInput) voiceoverEmotionInput.value = '';
        activeVoiceoverPlayer = null;
        activeVoiceoverTopic = '';
        voiceoverModal.classList.remove('visible');
    }

    function attachVoiceoverToPlayer(playerElement, blob, topic) {
        if (!playerElement || !blob) return;
        const container = playerElement.querySelector('.voiceover-container');
        if (!container) return;

        const previousAudio = container.querySelector('audio');
        if (previousAudio?.dataset?.objectUrl) {
            URL.revokeObjectURL(previousAudio.dataset.objectUrl);
        }

        const objectUrl = URL.createObjectURL(blob);
        const audioElement = document.createElement('audio');
        audioElement.controls = true;
        audioElement.src = objectUrl;
        audioElement.dataset.objectUrl = objectUrl;
        audioElement.setAttribute('preload', 'auto');

        container.innerHTML = '';
        container.appendChild(audioElement);
        container.classList.remove('empty', 'loading', 'error');
        container.classList.add('success');

        const downloadButton = playerElement.querySelector('.download-voiceover');
        if (downloadButton) {
            downloadButton.disabled = false;
        }
    }

    function handleFormSubmit(e) {
        e.preventDefault();
        const isInitial = e.currentTarget.id === 'initial-form';
        const submitButton = isInitial
            ? initialForm?.querySelector('button')
            : chatForm?.querySelector('button');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.classList.add('disabled');
        }
        const input = isInitial ? initialInput : chatInput;
        const topic = input.value.trim();
        if (!topic) return;

        if (isInitial) switchToChatView();

        conversationHistory.push({ role: 'user', content: topic });
        startGeneration(topic);
        input.value = '';
        if (isInitial) placeholderContainer?.classList?.remove('hidden');
    }

    async function startGeneration(topic) {
        console.log('Getting generation from backend.');
        appendUserMessage(topic);
        const agentThinkingMessage = appendAgentStatus(translations.agentThinking[currentLang]);
        const submitButton = document.querySelector('.submit-button');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.classList.add('disabled');
        }
        accumulatedCode = '';
        let inCodeBlock = false;
        let codeBlockElement = null;

        try {
            const response = await fetch(`${config.apiBaseUrl}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic: topic, history: conversationHistory })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;

                    const jsonStr = line.substring(6);
                    if (jsonStr.includes('[DONE]')) {
                        console.log('Streaming complete');
                        conversationHistory.push({ role: 'assistant', content: accumulatedCode });

                        if (!codeBlockElement) {
                            console.warn('No code block element created. Full response:', accumulatedCode);
                            throw new LLMParseError('LLM did not return a complete code block.');
                        }

                        if (!isHtmlContentValid(accumulatedCode)) {
                            console.warn('Invalid HTML received:\n', accumulatedCode);
                            throw new LLMParseError('Invalid HTML content received.');
                        }

                        markCodeAsComplete(codeBlockElement);

                        try {
                            if (accumulatedCode) {
                                appendAnimationPlayer(accumulatedCode, topic);
                            }
                        } catch (err) {
                            console.error('appendAnimationPlayer failed:', err);
                            throw new LLMParseError('Animation rendering failed.');
                        }
                        scrollToBottom();
                        return;
                    }

                    let data;
                    try {
                        data = JSON.parse(jsonStr);
                    } catch (err) {
                        console.error('Failed to parse JSON:', jsonStr);
                        throw new LLMParseError('Invalid response format from server.');
                    }

                    if (data.error) {
                        throw new LLMParseError(data.error);
                    }
                    const token = data.token || '';

                    if (!inCodeBlock && token.includes('```')) {
                        inCodeBlock = true;
                        if (agentThinkingMessage) agentThinkingMessage.remove();
                        codeBlockElement = appendCodeBlock();
                        const contentAfterMarker = token.substring(token.indexOf('```') + 3).replace(/^html\n/, '');
                        updateCodeBlock(codeBlockElement, contentAfterMarker);
                    } else if (inCodeBlock) {
                        if (token.includes('```')) {
                            inCodeBlock = false;
                            const contentBeforeMarker = token.substring(0, token.indexOf('```'));
                            updateCodeBlock(codeBlockElement, contentBeforeMarker);
                        } else {
                            updateCodeBlock(codeBlockElement, token);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Streaming failed:", error);
            if (agentThinkingMessage) agentThinkingMessage.remove();

            if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
                showWarning(translations.errorFetchFailed[currentLang]);
            } else if (error.message.includes('status: 429')) {
                showWarning(translations.errorTooManyRequests[currentLang]);
            } else if (error instanceof LLMParseError) {
                showWarning(translations.errorLLMParseError[currentLang]);
            } else {
                showWarning(translations.errorFetchFailed[currentLang]); // 默认 fallback
            }

            appendErrorMessage(translations.errorMessage[currentLang]);  // 保留 chat-log 中的提示
        } finally {
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.classList.remove('disabled');
        }
    }
    }

    function switchToChatView() {
        body.classList.remove('show-initial-view');
        body.classList.add('show-chat-view');
        languageSwitcher.style.display = 'none';
        document.getElementById('logo-chat').style.display = 'block';
    }

    function appendFromTemplate(template, text) {
        const node = template.content.cloneNode(true);
        const element = node.firstElementChild;
        if (text) element.innerHTML = element.innerHTML.replace('${text}', text);
        element.querySelectorAll('[data-translate-key]').forEach(el => {
            const key = el.dataset.translateKey;
            const translation = translations[key]?.[currentLang];
            if (translation) el.textContent = translation;
        });
        chatLog.appendChild(element);
        scrollToBottom();
        return element;
    }

    const appendUserMessage = (text) => appendFromTemplate(templates.user, text);
    const appendAgentStatus = (text) => appendFromTemplate(templates.status, text);
    const appendErrorMessage = (text) => appendFromTemplate(templates.error, text);
    const appendCodeBlock = () => appendFromTemplate(templates.code);

    function updateCodeBlock(codeBlockElement, text) {
        const codeElement = codeBlockElement.querySelector('code');
        if (!text || !codeElement) return;
        const span = document.createElement('span');
        span.textContent = text;
        codeElement.appendChild(span);
        accumulatedCode += text;

        const codeContent = codeElement.closest('.code-content');
        if (codeContent) {
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    codeContent.scrollTop = codeContent.scrollHeight;
                });
            });
        }
    }

    function markCodeAsComplete(codeBlockElement) {
        codeBlockElement.querySelector('[data-translate-key="generatingCode"]').textContent = translations.codeComplete[currentLang];
        codeBlockElement.querySelector('.code-details').removeAttribute('open');
    }

    function appendAnimationPlayer(htmlContent, topic) {
        console.log('Appending animation player with topic:', topic);
        const node = templates.player.content.cloneNode(true);
        const playerElement = node.firstElementChild;
        playerElement.querySelectorAll('[data-translate-key]').forEach(el => {
            const key = el.dataset.translateKey;
            el.textContent = translations[key]?.[currentLang] || el.textContent;
        });
        const iframe = playerElement.querySelector('.animation-iframe');
        iframe.srcdoc = htmlContent;

        playerElement.querySelector('.open-new-window').addEventListener('click', () => {
            const blob = new Blob([htmlContent], { type: 'text/html' });
            window.open(URL.createObjectURL(blob), '_blank');
        });
        playerElement.querySelector('.save-html').addEventListener('click', () => {
            const blob = new Blob([htmlContent], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            const a = Object.assign(document.createElement('a'), { href: url, download: `${topic.replace(/\s/g, '_') || 'animation'}.html` });
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(url);
            a.remove();
        });
        const voiceoverButton = playerElement.querySelector('.generate-voiceover');
        if (voiceoverButton) {
            voiceoverButton.disabled = true;
            voiceoverButton.classList.add('disabled');
            setVoiceoverButtonTranslation(voiceoverButton, 'voiceoverAutoGenerating');
        }
        const downloadVoiceoverButton = playerElement.querySelector('.download-voiceover');
        if (downloadVoiceoverButton) {
            downloadVoiceoverButton.disabled = true;
            downloadVoiceoverButton.addEventListener('click', () => {
                const audioElement = playerElement.querySelector('.voiceover-container audio');
                if (!audioElement) {
                    showWarning(translations.voiceoverMissingAudio[currentLang]);
                    return;
                }
                const url = audioElement.dataset.objectUrl || audioElement.src;
                const safeTopic = (topic || 'voiceover').trim().replace(/\s+/g, '_');
                const link = Object.assign(document.createElement('a'), {
                    href: url,
                    download: `${safeTopic || 'voiceover'}.wav`,
                });
                document.body.appendChild(link);
                link.click();
                link.remove();
            });
        }
        playerElement.querySelector('.export-video')?.addEventListener('click', () => {
            featureModal.querySelector('p').textContent = translations.featureComingSoon[currentLang];
            modalGitHubButton.textContent = translations.visitGitHub[currentLang];
            featureModal.classList.add('visible');
        });
        chatLog.appendChild(playerElement);
        scrollToBottom();
        autoGenerateVoiceoverForPlayer(playerElement, htmlContent, topic).catch((error) => {
            console.error('Auto voiceover generation encountered an unexpected error:', error);
        });
    }

    function isHtmlContentValid(htmlContent) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(htmlContent, "text/html");

        // 检查是否存在解析错误
        const parseErrors = doc.querySelectorAll("parsererror");
        if (parseErrors.length > 0) {
            console.warn("HTML 解析失败：", parseErrors[0].textContent);
            return false;
        }

        // 可选：检测是否有 <html><body> 结构或是否为空
        if (!doc.body || doc.body.innerHTML.trim() === "") {
            console.warn("HTML 内容为空");
            return false;
        }

        return true;
    }

    const scrollToBottom = () => chatLog.scrollTo({ top: chatLog.scrollHeight, behavior: 'smooth' });

    function setNextPlaceholder() {
        const placeholderTexts = translations.placeholders[currentLang];
        const newSpan = document.createElement('span');
        newSpan.textContent = placeholderTexts[placeholderIndex];
        placeholderContainer.innerHTML = '';
        placeholderContainer.appendChild(newSpan);
        placeholderIndex = (placeholderIndex + 1) % placeholderTexts.length;
    }

    function startPlaceholderAnimation() {
        if (placeholderInterval) clearInterval(placeholderInterval);
        const placeholderTexts = translations.placeholders[currentLang];
        if (placeholderTexts && placeholderTexts.length > 0) {
            placeholderIndex = 0;
            setNextPlaceholder();
            placeholderInterval = setInterval(setNextPlaceholder, 4000);
        }
    }

    function setLanguage(lang) {
        if (!['zh', 'en'].includes(lang)) return;
        currentLang = lang;
        document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
        document.querySelectorAll('[data-translate-key]').forEach(el => {
            const key = el.dataset.translateKey;
            const translation = translations[key]?.[lang];
            if (!translation) return;
            if (el.hasAttribute('placeholder')) el.placeholder = translation;
            else if (el.hasAttribute('title')) el.title = translation;
            else el.textContent = translation;
        });
        languageSwitcher.querySelectorAll('button').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === lang);
        });
        startPlaceholderAnimation();
        localStorage.setItem('preferredLanguage', lang);
    }

    let placeholderIndex = 0;

    function init() {
        initialInput.addEventListener('input', () => {
            placeholderContainer.classList.toggle('hidden', initialInput.value.length > 0);
        });
        initialInput.addEventListener('focus', () => clearInterval(placeholderInterval));
        initialInput.addEventListener('blur', () => {
            if (initialInput.value.length === 0) startPlaceholderAnimation();
        });

        initialForm.addEventListener('submit', handleFormSubmit);
        chatForm.addEventListener('submit', handleFormSubmit);
        newChatButton.addEventListener('click', () => location.reload());
        languageSwitcher.addEventListener('click', (e) => {
            const target = e.target.closest('button');
            if (target) setLanguage(target.dataset.lang);
        });

        function hideModal() {
            featureModal.classList.remove('visible');
        }

        modalCloseButton.addEventListener('click', hideModal);
        featureModal.addEventListener('click', (e) => {
            if (e.target === featureModal) hideModal();
        });

        modalGitHubButton.addEventListener('click', () => {
            window.open('https://github.com/fogsightai/fogsightai', '_blank');
            hideModal();
        });

        if (voiceoverCloseButton) {
            voiceoverCloseButton.addEventListener('click', (event) => {
                event.preventDefault();
                if (voiceoverSubmitButton?.disabled) return;
                closeVoiceoverModal();
            });
        }

        if (voiceoverModal) {
            voiceoverModal.addEventListener('click', (event) => {
                if (event.target === voiceoverModal && !voiceoverSubmitButton?.disabled) {
                    closeVoiceoverModal();
                }
            });
        }

        if (voiceoverForm) {
            voiceoverForm.addEventListener('submit', async (event) => {
                event.preventDefault();
                if (!activeVoiceoverPlayer) {
                    showWarning(translations.voiceoverServiceUnavailable[currentLang]);
                    return;
                }

                const narrationText = voiceoverTextInput?.value.trim();
                if (!narrationText) {
                    showWarning(translations.voiceoverTextRequired[currentLang]);
                    return;
                }

                const speakerFile = voiceoverSpeakerInput?.files?.[0];
                if (!speakerFile) {
                    showWarning(translations.voiceoverSpeakerRequired[currentLang]);
                    return;
                }

                resetVoiceoverStatus();
                if (voiceoverStatus) {
                    voiceoverStatus.textContent = translations.voiceoverGeneratingStatus[currentLang];
                }

                if (voiceoverSubmitButton) {
                    voiceoverSubmitButton.disabled = true;
                }

                const formData = new FormData();
                formData.append('text', narrationText);
                formData.append('speaker_audio', speakerFile);
                if (voiceoverEmotionInput?.files?.[0]) {
                    formData.append('emo_audio', voiceoverEmotionInput.files[0]);
                }

                try {
                    const response = await fetch(`${config.apiBaseUrl}/voiceover`, {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        const errorText = await response.text();
                        let message = translations.voiceoverErrorStatus[currentLang];
                        try {
                            const parsed = JSON.parse(errorText);
                            if (parsed?.detail) message = parsed.detail;
                        } catch (parseError) {
                            console.warn('Failed to parse voiceover error response:', parseError);
                        }
                        if (voiceoverStatus) {
                            voiceoverStatus.textContent = message;
                            voiceoverStatus.classList.add('error');
                        }
                        showWarning(message);
                        return;
                    }

                    const blob = await response.blob();
                    attachVoiceoverToPlayer(activeVoiceoverPlayer, blob, activeVoiceoverTopic || narrationText);
                    if (voiceoverStatus) {
                        voiceoverStatus.textContent = translations.voiceoverSuccessStatus[currentLang];
                        voiceoverStatus.classList.add('success');
                    }

                    setTimeout(() => {
                        closeVoiceoverModal();
                    }, 600);
                } catch (error) {
                    console.error('Voiceover generation failed:', error);
                    if (voiceoverStatus) {
                        voiceoverStatus.textContent = translations.voiceoverErrorStatus[currentLang];
                        voiceoverStatus.classList.add('error');
                    }
                    showWarning(translations.voiceoverServiceUnavailable[currentLang]);
                } finally {
                    if (voiceoverSubmitButton) {
                        voiceoverSubmitButton.disabled = false;
                    }
                }
            });
        }

        const savedLang = localStorage.getItem('preferredLanguage');
        const browserLang = navigator.language?.toLowerCase() || ''; // e.g. 'zh-cn'

        let initialLang = 'en';
        if (['zh', 'en'].includes(savedLang)) {
            initialLang = savedLang;
        } else if (browserLang.startsWith('zh')) {
            initialLang = 'zh';
        } else if (browserLang.startsWith('en')) {
            initialLang = 'en';
        }

        setLanguage(initialLang);
    }

    init();
});

function showWarning(message) {
    const box = document.getElementById('warning-box');
    const overlay = document.getElementById('overlay');
    const text = document.getElementById('warning-message');
    text.textContent = message;
    box.style.display = 'flex';
    overlay.style.display = 'block';

    setTimeout(() => {
        hideWarning();
    }, 10000);
}

function hideWarning() {
    document.getElementById('warning-box').style.display = 'none';
    document.getElementById('overlay').style.display = 'none';
}
