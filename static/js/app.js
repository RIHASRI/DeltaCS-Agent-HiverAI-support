document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('query-input');
    const btnSubmit = document.getElementById('btn-submit');
    const btnRunBenchmark = document.getElementById('btn-run-benchmark');
    const chatFeed = document.getElementById('chat-feed');
    
    btnSubmit.addEventListener('click', analyzeQuery);
    btnRunBenchmark.addEventListener('click', showBenchmarkModal);
    
    // Auto submit on Enter (Ctrl+Enter)
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            analyzeQuery();
        }
    });
});

function setQuery(text) {
    document.getElementById('query-input').value = text;
    analyzeQuery();
}

async function analyzeQuery() {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;
    
    const complexity = document.getElementById('sel-complexity').value;
    const urgency = document.getElementById('sel-urgency').value;
    const sentiment = document.getElementById('sel-sentiment').value;
    
    const chatFeed = document.getElementById('chat-feed');
    
    // Clear placeholder if present
    const placeholder = chatFeed.querySelector('.chat-placeholder');
    if (placeholder) placeholder.remove();
    
    // Append Customer Tweet
    const customerMsg = document.createElement('div');
    customerMsg.className = 'chat-msg customer';
    customerMsg.innerHTML = `
        <div class="msg-header"><i class="fa-brands fa-twitter"></i> Customer Tweet</div>
        <div class="msg-bubble">${escapeHtml(query)}</div>
    `;
    chatFeed.appendChild(customerMsg);
    chatFeed.scrollTop = chatFeed.scrollHeight;
    
    // Show Loading Agent Message
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'chat-msg agent';
    loadingMsg.innerHTML = `
        <div class="msg-header"><i class="fa-solid fa-robot"></i> @Delta AI Agent</div>
        <div class="msg-bubble"><i class="fa-solid fa-circle-notch fa-spin"></i> Processing query and evaluating safety rules...</div>
    `;
    chatFeed.appendChild(loadingMsg);
    chatFeed.scrollTop = chatFeed.scrollHeight;
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, complexity, urgency, sentiment })
        });
        
        const data = await response.json();
        
        // Update Agent Reply Bubble
        loadingMsg.innerHTML = `
            <div class="msg-header"><i class="fa-solid fa-robot"></i> @Delta AI Agent</div>
            <div class="msg-bubble">${escapeHtml(data.generated_reply)}</div>
        `;
        chatFeed.scrollTop = chatFeed.scrollHeight;
        
        // Update Inspector Panel
        updateInspector(data);
        
    } catch (err) {
        console.error('API Error:', err);
        loadingMsg.innerHTML = `
            <div class="msg-header"><i class="fa-solid fa-robot"></i> @Delta AI Agent</div>
            <div class="msg-bubble" style="border-color: red;">Error processing request. Check server console.</div>
        `;
    }
}

function updateInspector(data) {
    // 1. Badge Intent
    const badge = document.getElementById('insp-intent-badge');
    badge.textContent = `${data.intent} (${(data.confidence * 100).toFixed(1)}%)`;
    
    // 2. Entities
    const entContainer = document.getElementById('entity-container');
    entContainer.innerHTML = '';
    
    let hasEntities = false;
    for (const [key, val] of Object.entries(data.entities)) {
        if (val) {
            hasEntities = true;
            const tag = document.createElement('span');
            tag.className = 'entity-tag';
            tag.textContent = `${key.toUpperCase()}: ${val}`;
            entContainer.appendChild(tag);
        }
    }
    if (!hasEntities) {
        entContainer.innerHTML = '<span class="entity-tag placeholder">No specific entity identifiers extracted</span>';
    }
    
    // 3. Probability Bars
    const barsContainer = document.getElementById('intent-bars-container');
    barsContainer.innerHTML = '';
    
    const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    for (const [intentName, prob] of sortedProbs) {
        const pct = (prob * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'intent-bar-row';
        row.innerHTML = `
            <span class="intent-name">${intentName}</span>
            <div class="bar-track"><div class="bar-fill" style="width: ${pct}%;"></div></div>
            <span class="intent-pct">${pct}%</span>
        `;
        barsContainer.appendChild(row);
    }
    
    // 4. Escalation Card
    const escBadge = document.getElementById('esc-decision-badge');
    const riskText = document.getElementById('esc-risk-text');
    const riskBar = document.getElementById('risk-bar');
    const reasonsList = document.getElementById('esc-reasons-list');
    
    const esc = data.escalation;
    escBadge.textContent = esc.decision;
    if (esc.decision === 'HUMAN_ESCALATION') {
        escBadge.className = 'esc-decision badge-human';
    } else {
        escBadge.className = 'esc-decision badge-auto';
    }
    
    riskText.textContent = `Risk Score: ${esc.risk_score.toFixed(2)}`;
    riskBar.style.width = `${(esc.risk_score * 100).toFixed(0)}%`;
    if (esc.risk_score > 0.45) {
        riskBar.style.backgroundColor = '#e31837';
    } else {
        riskBar.style.backgroundColor = '#00c853';
    }
    
    reasonsList.innerHTML = '';
    for (const r of esc.reasons) {
        const li = document.createElement('li');
        li.textContent = r;
        reasonsList.appendChild(li);
    }
    
    // 5. LLM Rubric Grid
    const judge = data.judge_eval;
    document.getElementById('r-rel').textContent = judge.relevance.toFixed(1);
    document.getElementById('r-tone').textContent = judge.tone_empathy.toFixed(1);
    document.getElementById('r-act').textContent = judge.actionability.toFixed(1);
    document.getElementById('r-safe').textContent = judge.policy_safety.toFixed(1);
}

async function showBenchmarkModal() {
    const modal = document.getElementById('modal-benchmark');
    const modalBody = document.getElementById('modal-body');
    modal.style.display = 'flex';
    modalBody.innerHTML = '<div class="loading-spinner"><i class="fa-solid fa-circle-notch fa-spin"></i> Loading benchmark metrics...</div>';
    
    try {
        const res = await fetch('/api/benchmark');
        const report = await res.json();
        
        modalBody.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 16px;">
                <h3>Benchmark Overview (${report.total_benchmark_samples} Golden Samples)</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                    <div style="background:#0b1329; padding:12px; border-radius:8px;">
                        <span style="font-size:11px; color:#8a99ad;">Accuracy</span>
                        <div style="font-size:20px; font-weight:bold;">${(report.intent_classification.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div style="background:#0b1329; padding:12px; border-radius:8px;">
                        <span style="font-size:11px; color:#8a99ad;">Macro F1</span>
                        <div style="font-size:20px; font-weight:bold;">${report.intent_classification.macro_f1.toFixed(3)}</div>
                    </div>
                    <div style="background:#0b1329; padding:12px; border-radius:8px;">
                        <span style="font-size:11px; color:#8a99ad;">LLM Judge Score</span>
                        <div style="font-size:20px; font-weight:bold;">${report.llm_judge_metrics.mean_overall_score.toFixed(2)} / 5.0</div>
                    </div>
                </div>
                
                <h4>Text Quality Metrics</h4>
                <p>Mean ROUGE-L: <b>${report.text_quality_metrics.mean_rouge_l.toFixed(4)}</b> | Mean BLEU-1: <b>${report.text_quality_metrics.mean_bleu_1.toFixed(4)}</b></p>
                
                <h4>Inter-Rater Agreement</h4>
                <p>Pearson r: <b>${report.inter_rater_agreement.pearson_r.toFixed(4)}</b> | MAE: <b>${report.inter_rater_agreement.mean_absolute_error.toFixed(4)}</b> | Kappa Proxy: <b>${report.inter_rater_agreement.cohens_kappa_proxy.toFixed(4)}</b></p>
            </div>
        `;
    } catch (err) {
        modalBody.innerHTML = '<div style="color:red;">Failed to load benchmark metrics.</div>';
    }
}

function closeModal() {
    document.getElementById('modal-benchmark').style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
