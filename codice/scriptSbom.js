const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const axios = require('axios');
const coloriGravità = {
        HIGH: '\x1b[31m', // Rosso
        MODERATE: '\x1b[33m', // Giallo
        LOW: '\x1b[32m', // Verde
};

// Inizializzazione di default per `logSoloFile` all'inizio del file
let logSoloFile = (...args) => {
    // Funzione di fallback che non genera errori
    require('fs').appendFileSync('default-log.txt', args.join(' ') + '\n');
};
/**
 * Trova i file principali (package.json, yarn.lock) solo nella directory principale.
 */
function trovaFilePrincipali(projectDir) {
    const principali = [];
    ['package.json', 'yarn.lock'].forEach((file) => {
        const filePath = path.join(projectDir, file);
        if (fs.existsSync(filePath)) {
            principali.push(filePath);
        }
    });
    return principali;
}

/**
 * Scrive il report delle vulnerabilità in un file Markdown.
 */

function scriviReportVulnerabilità(vulnerabilità, projectTimestampDir, projectName) {
    const reportPath = path.join(projectTimestampDir, `vulnerability-report.md`);

    // Calcolo della gravità massima e del riepilogo per gravità
    const riepilogo = {
        CRITICAL: 0,
        HIGH: 0,
        MODERATE: 0,
        LOW: 0,
    };

    vulnerabilità.forEach((vuln) => {
        vuln.vulnerabilities.forEach((v) => {
            if (riepilogo[v.severity] !== undefined) {
                riepilogo[v.severity]++;
            }
        });
    });

    const gravitàMassima = Math.max(
        ...Object.entries(riepilogo).map(([severity, count]) => {
            if (severity === 'CRITICAL') return count > 0 ? 3 : 0;
            if (severity === 'HIGH') return count > 0 ? 2 : 0;
            if (severity === 'MODERATE') return count > 0 ? 1 : 0;
            return 0;
        })
    );

    const lines = [
        `# Report di Vulnerabilità`,
        `## Progetto: ${projectName}`,
        `### Data: ${new Date().toISOString()}`,
        `### Gravità massima rilevata: ${gravitàMassima === 3 ? 'CRITICAL' : gravitàMassima === 2 ? 'HIGH' : 'MODERATE'}\n`,
    ];

    // Riepilogo delle vulnerabilità per gravità
    lines.push(`### Dettaglio per gravità:`);
    Object.entries(riepilogo).forEach(([severity, count]) => {
        lines.push(`- **${severity}**: ${count}`);
    });
    lines.push(""); // Linea vuota per separare

    // Dettaglio delle vulnerabilità
    if (vulnerabilità.length === 0) {
        lines.push("Nessuna vulnerabilità trovata.\n");
    } else {
        const totaleVulnerabilità = vulnerabilità.reduce((acc, vuln) => acc + vuln.vulnerabilities.length, 0);

        lines.push(`Numero di pacchetti vulnerabili trovati: ${vulnerabilità.length}`);
        lines.push(`Numero totale di vulnerabilità: ${totaleVulnerabilità}\n`);

        vulnerabilità.forEach((vuln) => {
            lines.push(`### Pacchetto: ${vuln.package}@${vuln.version}`);
            vuln.vulnerabilities.forEach((detail) => {
                lines.push(`- **ID**: ${detail.id}`);
                lines.push(`  - **Descrizione**: ${detail.summary}`);
                lines.push(`  - **Gravità**: ${detail.severity}`);
                lines.push(`  - **Versione Fissata**: ${detail.fixedVersion || 'N/A'}`);
                if (detail.references.length > 0) {
                    lines.push("  - **Riferimenti**:");
                    detail.references.forEach((ref) => {
                        lines.push(`    - ${ref.url}`);
                    });
                }
                lines.push(""); // Linea vuota per separare
            });
        });
    }


    // Scrittura del file
    fs.writeFileSync(reportPath, lines.join("\n"));
    console.log(`[+] Report delle vulnerabilità generato in: ${reportPath}`);
}



function installaDipendenze(projectDir) {
    console.log("[*] Verifica e installazione delle dipendenze nella directory principale...");
    const principali = trovaFilePrincipali(projectDir);

    principali.forEach((file) => {
        const directory = path.dirname(file);
        if (directory !== projectDir) {
            logSoloFile(`[!] Saltata directory non root: ${directory}`);
            return;
        }

        if (file.includes('package.json')) {
            try {
                console.log(`[*] Sto installando le dipendenze dal file package.json...`);
                execSync('npm install --legacy-peer-deps > npm-install.log 2>&1', { cwd: directory });
            } catch (err) {
                logSoloFile(`[!] Errore durante l'installazione delle dipendenze con npm in ${directory}: ${err.message}`);
                logSoloFile(`[!] Continuo con l'analisi delle dipendenze nonostante l'errore.`);
            }
        } else if (file.includes('yarn.lock')) {
            try {
                console.log(`[*] Sto installando le dipendenze dal file yarn.lock...`);
                execSync('yarn install > yarn-install.log 2>&1', { cwd: directory });
            } catch (err) {
                logSoloFile(`[!] Errore durante l'installazione con Yarn in ${directory}: ${err.message}`);
                logSoloFile(`[!] Continuo con l'analisi delle dipendenze nonostante l'errore.`);
            }
        }
    });
}
/**
 * Analizza le dipendenze dalla directory principale leggendo il file package.json.
 */
function analizzaDipendenze(projectDir) {
    const packageFile = path.join(projectDir, 'package.json');
    const dependencies = [];

    if (!fs.existsSync(packageFile)) {
        console.log(`[!] package.json non trovato in ${projectDir}`);
        return dependencies;
    }

    const data = JSON.parse(fs.readFileSync(packageFile, 'utf8'));
    ['dependencies', 'devDependencies', 'peerDependencies', 'optionalDependencies'].forEach((depType) => {
        if (data[depType]) {
            for (const [dep, version] of Object.entries(data[depType])) {
                const cleanVersion = version.replace(/^[~^<>=]+/, '');
                dependencies.push(`${dep}@${cleanVersion}`);
            }
        }
    });

    return dependencies;
}

/**
 * Filtra dipendenze non valide o ridondanti.
 */
function filtraDipendenzeInvalidi(dependencies) {
    const validDependencies = [];
    const seenDependencies = new Set();

    dependencies.forEach((dep) => {
        if (typeof dep === 'string' && dep.includes('@') && !seenDependencies.has(dep)) {
            const [depName, depVersion] = dep.split('@', 2);

            if (!depName || !depVersion || ['*', 'latest'].includes(depVersion)) {
                logSoloFile(`[!] Dipendenza ignorata: ${dep}. Motivo: Nome o versione non specificati o non accettabili.`);
                return;
            }

            const versionPattern = /^\d+\.\d+\.\d+(-\w+\.\d+)?$/;
            if (!versionPattern.test(depVersion)) {
                logSoloFile(`[!] Dipendenza ignorata: ${dep}. Motivo: Formato versione non valido.`);
                return;
            }

            validDependencies.push(dep);
            seenDependencies.add(dep);
        } else {
            logSoloFile(`[!] Dipendenza non valida ignorata: ${dep}.`);
        }
    });

    return validDependencies;
}

async function cercaVersioneAggiornata(depName) {
    try {
        const response = await axios.get(`https://registry.npmjs.org/${depName}`);
        const latestVersion = response.data['dist-tags'].latest;
        console.log(`[INFO] Versione più recente per ${depName}: ${latestVersion}`);
        return latestVersion;
    } catch (err) {
        console.error(`[!] Impossibile trovare una versione aggiornata per ${depName}: ${err.message}`);
        return null;
    }
}

async function aggiornaDipendenze(dependencies) {
    console.log("[*] Verifica versioni più recenti per le dipendenze...");
    logSoloFile("[*] Verifica versioni più recenti per le dipendenze...");

    const dipendenzeDaAggiornare = [];

    for (let i = 0; i < dependencies.length; i++) {
        const [name, version] = dependencies[i].split('@');
        const latestVersion = await cercaVersioneAggiornata(name);

        if (latestVersion && latestVersion !== version) {
            dipendenzeDaAggiornare.push({
                name,
                currentVersion: version,
                latestVersion,
            });

            // Solo nel log
            logSoloFile(`[INFO] Versione più recente per ${name}: ${latestVersion}`);
        }
    }

    logSoloFile(`[INFO] Fine verifica versioni più recenti.`);
    return dipendenzeDaAggiornare;
}

async function aggiornaDipendenzeAutomaticamente(dipendenzeDaAggiornare, projectDir) {
    console.log("[INFO] Avvio aggiornamento automatico delle dipendenze...");
    for (const dep of dipendenzeDaAggiornare) {
        try {
            console.log(`[INFO] Aggiornamento ${dep.name} da ${dep.currentVersion} a ${dep.latestVersion}...`);

            // Cattura stdout e stderr durante l'esecuzione del comando
            const command = `npm install ${dep.name}@${dep.latestVersion} --save`;
            const options = { cwd: projectDir, stdio: ['inherit', 'pipe', 'pipe'] };

            const result = execSync(command, options);

            // Scrivi l'output di successo nel file di log
            logSoloFile(`[INFO] Output aggiornamento per ${dep.name}:`);
            logSoloFile(result.toString());
        } catch (error) {
            // Scrivi l'errore nel file di log
            const errorMessage = `[!] Errore durante l'aggiornamento di ${dep.name}: ${error.message}`;
            logSoloFile(errorMessage);

            if (error.stderr) {
                logSoloFile(`[!] Dettagli dell'errore per ${dep.name}:`);
                logSoloFile(error.stderr.toString());
            }
        }
    }
    console.log("[INFO] Aggiornamento completato.");
}


/**
 * Ottimizza le informazioni sulle vulnerabilità, includendo versioni sicure suggerite.
 */
function ottimizzaVulnerabilita(components) {
    return components.map((comp) => {
        if (comp.vulnerabilities && comp.vulnerabilities.length > 0) {
            comp.vulnerabilities = comp.vulnerabilities.map((vuln) => {
                if (vuln.ranges && vuln.ranges.length > 0) {
                    const fixedRange = vuln.ranges.find((range) => range.fixed);
                    vuln.fixedVersion = fixedRange ? fixedRange.fixed : 'NON DISPONIBILE';
                } else {
                    vuln.fixedVersion = 'NON DISPONIBILE';
                }
                return vuln;
            });
        }
        return comp;
    });
}

/**
 * Scrive uno SBoM in formato JSON in una directory personalizzata con sottodirectory basate su progetto e timestamp.
*/

async function generaSbom(projectTimestampDir, projectName, dependencies, vulnerabilità) {
    console.log("[*] Generazione dello SBoM in formato JSON...");

    const sbomPath = path.join(projectTimestampDir, `sbom.json`);
    const sbom = {
        bomFormat: 'CycloneDX',
        specVersion: '1.4',
        summary: {
            totalComponents: dependencies.length,
            totalVulnerabilities: vulnerabilità.length,
        },
        components: await Promise.all(
            dependencies.map(async (dep) => {
                const [name, version] = dep.split('@');
                const vulnDetails = vulnerabilità.find((v) => v.package === name && v.version === version);

                // Calcolo della licenza
                const license = await ottieniLicenza(name);

                return {
                    type: 'library',
                    name,
                    version,
                    license,
                    purl: `pkg:npm/${name}@${version}`,
                    vulnerabilities: vulnDetails
                        ? vulnDetails.vulnerabilities.map((v) => ({
                              id: v.id,
                              summary: v.summary,
                              severity: v.severity || 'UNKNOWN',
                              references: v.references || [],
                              fixedVersion: v.ranges
                                  ? v.ranges.find((r) => r.fixed)?.fixed || 'N/A'
                                  : 'N/A',
                          }))
                        : [],
                    repository: `https://www.npmjs.com/package/${name}`,
                };
            })
        ),
    };

    fs.writeFileSync(sbomPath, JSON.stringify(sbom, null, 2));
    console.log(`[+] SBoM generato in: ${sbomPath}`);
    return sbomPath;
}

async function verificaConGitHubAdvisory(depName, depVersion) {
    const GITHUB_API_URL = "https://api.github.com/graphql";
    const GITHUB_TOKEN = process.env.GITHUB_TOKEN;

    if (!GITHUB_TOKEN) {
        console.error("[!] GITHUB_TOKEN non configurato.");
        return [];
    }

    const query = `
        query($name: String!) {
            securityVulnerabilities(package: $name, first: 10) {
                edges {
                    node {
                        advisory {
                            summary
                            severity
                            references { url }
                        }
                    }
                }
            }
        }
    `;

    try {
        const response = await axios.post(
            GITHUB_API_URL,
            { query, variables: { name: depName } },
            { headers: { Authorization: `Bearer ${GITHUB_TOKEN}` } }
        );

        logSoloFile(`[DEBUG] Risposta GitHub per ${depName}@${depVersion}:`, response.data);
        return response.data.data.securityVulnerabilities.edges.map((e) => e.node.advisory);
    } catch (err) {
        logSoloFile(`[!] Errore GitHub Advisory per ${depName}@${depVersion}: ${err.message}`);
        return [];
    }
}

async function cercaVulnerabilita(dependencies) {
    console.log("[*] Verifica delle vulnerabilità su OSV.dev e GitHub Advisory...");
    logSoloFile("[*] Verifica delle vulnerabilità su OSV.dev e GitHub Advisory...");

    const results = [];

    for (const dep of dependencies) {
        const [name, version] = dep.split('@');

        if (!name || !version || version.includes('*') || version.includes('latest')) continue;

        const payload = {
            version,
            package: { name, ecosystem: "npm" },
        };

        try {
            const osvResponse = await axios.post('https://api.osv.dev/v1/query', payload);
            const osvVulns = osvResponse.data.vulns || [];
            logSoloFile(`[DEBUG] OSV.dev per ${name}@${version}:`, osvResponse.data);

            const githubVulns = await verificaConGitHubAdvisory(name, version);
            logSoloFile(`[DEBUG] GitHub Advisory per ${name}@${version}:`, githubVulns);

            const combinedVulns = [
                ...osvVulns.map((v) => ({
                    id: v.id,
                    summary: v.summary,
                    severity: v.database_specific?.severity || v.severity || 'UNKNOWN',
                    references: v.references || [],
                })),
                ...githubVulns.map((v) => ({
                    id: `GitHub-${v.id || 'UNKNOWN'}`,
                    summary: v.summary || 'No summary provided',
                    severity: v.severity || 'UNKNOWN',
                    references: v.references || [],
                })),
            ];

            if (combinedVulns.length > 0) {
                results.push({ package: name, version, vulnerabilities: combinedVulns });
            }
        } catch (err) {
            logSoloFile(`[!] Errore durante la verifica di ${dep}: ${err.message}`);
        }
    }

    if (results.length > 0) {
        console.log(`[INFO] Vulnerabilità trovate: ${results.length}`);
    } else {
        console.log("[INFO] Nessuna vulnerabilità rilevata.");
    }

    logSoloFile(`[INFO] Dettagli vulnerabilità completati.`);
    return results;
}


async function ottieniLicenza(depName) {
    try {
        const response = await axios.get(`https://registry.npmjs.org/${depName}`);
        const license = response.data.license || 'UNKNOWN';
        return license;
    } catch (err) {
        console.error(`[!] Errore nel recupero della licenza per ${depName}: ${err.message}`);
        return 'UNKNOWN';
    }
}

function verificaCoerenzaVulnerabilità(vulnerabilità, riepilogoConsole) {
    const riepilogoFile = {
        CRITICAL: 0,
        HIGH: 0,
        MODERATE: 0,
        LOW: 0,
    };

    vulnerabilità.forEach((vuln) => {
        vuln.vulnerabilities.forEach((v) => {
            if (riepilogoFile[v.severity] !== undefined) {
                riepilogoFile[v.severity]++;
            }
        });
    });

    Object.entries(riepilogoFile).forEach(([severity, count]) => {
        if (count !== (riepilogoConsole[severity] || 0)) {
            console.warn(`[!] Discrepanza rilevata per ${severity}: File = ${count}, Console = ${riepilogoConsole[severity] || 0}`);
        }
    });
}

/**
 * Funzione principale.
 */
async function main() {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.error("[!] Per favore specifica la directory del progetto come primo argomento.");
        process.exit(1);
    }

    if (!process.env.GITHUB_TOKEN) {
        console.error("[!] GITHUB_TOKEN non configurato. Assicurati di esportarlo come variabile d'ambiente.");
        process.exit(1);
    }

    const projectDir = path.resolve(args[0]);
    const projectName = path.basename(projectDir);
    const baseOutputDir = path.join(process.env.HOME, 'Scrivania', 'SBOM', 'codice', 'risultati');

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const projectTimestampDir = path.join(baseOutputDir, projectName, timestamp);

    if (!fs.existsSync(projectTimestampDir)) {
        fs.mkdirSync(projectTimestampDir, { recursive: true });
        console.log(`[+] Directory di output creata: ${projectTimestampDir}`);
    }

    const consoleLogPath = path.join(projectTimestampDir, 'console-output.txt');
    const outputFile = fs.createWriteStream(consoleLogPath);

    // Configura correttamente `logSoloFile` per scrivere su un file specifico
    logSoloFile = (...args) => {
        outputFile.write(args.join(' ') + '\n');
    };

    const originalConsoleLog = console.log;

    // Filtra i messaggi per console
    console.log = (...args) => {
        const message = args.join(' ');
        if (
            message.startsWith("[INFO] Dettagli vulnerabilità completati") ||
            message.startsWith("[+] Report delle vulnerabilità generato in:") ||
            message.startsWith("[*] Generazione dello SBoM in formato JSON") ||
            message.startsWith("[+] SBoM generato in:") ||
            message.startsWith("[*] Analisi completata con successo") ||
            message.startsWith("[?] Vuoi aggiornare automaticamente le dipendenze proposte?") ||
            message.startsWith("[INFO] Riepilogo:") ||
            message.startsWith("- Totale componenti analizzati:") ||
            message.startsWith("- Totale vulnerabilità trovate:")
        ) {
            originalConsoleLog(...args);
        } else {
            outputFile.write(message + '\n');
        }
    };

    try {
        console.log(`[INFO] Analisi avviata per il progetto: ${projectName}`);
        console.log(`[INFO] Directory del progetto: ${projectDir}`);
        console.log(`[INFO] Directory di output: ${projectTimestampDir}`);

        installaDipendenze(projectDir);

        let dependencies = analizzaDipendenze(projectDir);
        dependencies = filtraDipendenzeInvalidi(dependencies);

        const dipendenzeDaAggiornare = await aggiornaDipendenze(dependencies);

        const vulnerabilità = await cercaVulnerabilita(dependencies);

        scriviReportVulnerabilità(vulnerabilità, projectTimestampDir, projectName);
        await generaSbom(projectTimestampDir, projectName, dependencies, vulnerabilità);

        console.log("[*] Analisi completata con successo.");

        console.log(`[INFO] Riepilogo:`);
        console.log(`- Totale componenti analizzati: ${dependencies.length}`);
        console.log(`- Totale vulnerabilità trovate: ${vulnerabilità.length}`);


        if (dipendenzeDaAggiornare.length > 0) {
            console.log(`[?] Vuoi aggiornare automaticamente le dipendenze proposte? (y/n)`);
            process.stdin.once('data', async (data) => {
                const risposta = data.toString().trim().toLowerCase();
                if (risposta === 'y') {
                    await aggiornaDipendenzeAutomaticamente(dipendenzeDaAggiornare, projectDir);
                } else {
                    console.log("[INFO] Aggiornamento delle dipendenze annullato dall'utente.");
                }
                process.exit(0);
            });
        }
    } catch (error) {
        console.error("[!] Errore durante l'analisi:", error.message);
    } finally {
        outputFile.end();
        console.log = originalConsoleLog;
    }
}

main();

