const STANDARD_WARNING = "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems.";
const SAMPLE_APPLICATION = { brand_name:"Acme Spirits", class_type:"Kentucky Straight Bourbon Whiskey", alcohol_content:"45% Alc. by Vol.", net_contents:"750 mL", warning_statement:STANDARD_WARNING };
const MAX_FILE_SIZE = 25 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png","image/jpeg","image/webp","image/gif"]);
const FIELD_LABELS = { brand_name:"Brand name", class_type:"Class or type", alcohol_content:"Alcohol content", net_contents:"Net contents", warning_statement:"Government warning" };
const STATUS_LABELS = { pass:"Pass", fail:"Fail", needs_review:"Needs review" };
const byId = (id) => document.getElementById(id);
const form=byId("verifyForm"), fileInput=byId("fileInput"), uploadArea=byId("uploadArea"), filePreview=byId("filePreview"), previewImage=byId("previewImage"), fileName=byId("fileName"), fileSize=byId("fileSize"), extractionMode=byId("extractionMode"), modeHelp=byId("modeHelp"), applicationSection=byId("applicationSection"), compareWorkflowButton=byId("compareWorkflowButton"), scanWorkflowButton=byId("scanWorkflowButton"), workflowHelp=byId("workflowHelp"), actionHelp=byId("actionHelp"), submitButton=byId("submitButton"), buttonLabel=submitButton.querySelector(".button-label"), serviceStatus=byId("serviceStatus"), loading=byId("loading"), loadingHeading=byId("loadingHeading"), loadingElapsed=byId("loadingElapsed"), loadingMessage=byId("loadingMessage"), errorBanner=byId("errorBanner"), errorMessage=byId("errorMessage"), noticeBanner=byId("noticeBanner"), noticeMessage=byId("noticeMessage"), resultsSection=byId("resultsSection"), resultSummary=byId("resultSummary"), summaryLabel=byId("summaryLabel"), resultsHeading=byId("resultsHeading"), resultCounts=byId("resultCounts"), summaryIcon=byId("summaryIcon"), scanAdvisory=byId("scanAdvisory"), scanAdvisoryText=byId("scanAdvisoryText"), fieldResultsHeading=byId("fieldResultsHeading"), fieldResultsHelp=byId("fieldResultsHelp"), serverTime=byId("serverTime"), roundTripTime=byId("roundTripTime"), resultFields=byId("resultFields");
let selectedFiles=[], previewUrl=null, timerId=null, serviceReady=false, localAvailable=true, visionAvailable=false, visionProvider=null, workflowMode="compare";

// The HTML contains the same default so it is visible before JavaScript loads.
// Keep this fallback for older cached markup and leave the field editable.
const warningStatement = byId("warningStatement");
if (!warningStatement.value.trim()) warningStatement.value = STANDARD_WARNING;

function formatDuration(ms){if(!Number.isFinite(ms))return "Not available";return ms<1000?`${Math.round(ms)} ms`:`${(ms/1000).toFixed(2)} seconds`}
function formatFileSize(bytes){return bytes<1048576?`${Math.max(1,Math.round(bytes/1024))} KB`:`${(bytes/1048576).toFixed(1)} MB`}
function setNotice(message){noticeMessage.textContent=message;noticeBanner.classList.toggle("hidden",!message)}
function hideError(){errorBanner.classList.add("hidden");errorMessage.textContent=""}
function showError(message,requestTimeMs=null){const timing=Number.isFinite(requestTimeMs)?` Request time: ${formatDuration(requestTimeMs)}.`:"";errorMessage.textContent=`${message}${timing}`;errorBanner.classList.remove("hidden");errorBanner.focus({preventScroll:true});errorBanner.scrollIntoView({behavior:"smooth",block:"center"})}
function clearPreviewUrl(){if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null}
function submitLabel(){return workflowMode==="scan"?"Scan label":selectedFiles.length>1?"Verify labels":"Verify label"}
function clearSelectedFiles(){selectedFiles=[];fileInput.value="";clearPreviewUrl();previewImage.removeAttribute("src");filePreview.classList.add("hidden");uploadArea.classList.remove("hidden","is-dragging");buttonLabel.textContent=submitLabel()}
function selectFiles(files){
  hideError();let candidates=Array.from(files||[]);if(!candidates.length)return;if(workflowMode==="scan"&&candidates.length>1){candidates=[candidates[0]];setNotice("Scan-only mode reads one label at a time. Only the first selected image was added.")}
  const invalid=candidates.find(file=>!ALLOWED_TYPES.has(file.type)), oversized=candidates.find(file=>file.size>MAX_FILE_SIZE);
  if(invalid){clearSelectedFiles();showError(`${invalid.name} is not a supported image. Choose PNG, JPEG, WEBP, or GIF.`);return}
  if(oversized){clearSelectedFiles();showError(`${oversized.name} exceeds the 25 MB upload limit.`);return}
  selectedFiles=candidates;clearPreviewUrl();previewUrl=URL.createObjectURL(candidates[0]);previewImage.src=previewUrl;
  fileName.textContent=candidates.length===1?candidates[0].name:`${candidates.length} label images selected`;
  const total=candidates.reduce((sum,file)=>sum+file.size,0);fileSize.textContent=candidates.length===1?`${formatFileSize(total)} · ${candidates[0].type}`:`${formatFileSize(total)} total · batch verification`;
  uploadArea.classList.add("hidden");filePreview.classList.remove("hidden");buttonLabel.textContent=submitLabel();
}
function applicationPayload(){return {brand_name:byId("brandName").value.trim(),class_type:byId("classType").value.trim(),alcohol_content:byId("alcoholContent").value.trim(),net_contents:byId("netContents").value.trim(),warning_statement:byId("warningStatement").value.trim()}}
async function optimizeLocalUpload(file){
  if(file.type==="image/jpeg"||file.type==="image/gif"||file.size<300000||typeof createImageBitmap!=="function")return file;
  try{
    const bitmap=await createImageBitmap(file),maxSide=2000,scale=Math.min(1,maxSide/Math.max(bitmap.width,bitmap.height)),canvas=document.createElement("canvas");
    canvas.width=Math.max(1,Math.round(bitmap.width*scale));canvas.height=Math.max(1,Math.round(bitmap.height*scale));const context=canvas.getContext("2d");context.fillStyle="#fff";context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(bitmap,0,0,canvas.width,canvas.height);bitmap.close();
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,"image/jpeg",0.94));
    return blob&&blob.size<file.size?new File([blob],file.name,{type:"image/jpeg",lastModified:file.lastModified}):file;
  }catch{return file}
}
function clearApplicationFields(){for(const name of Object.keys(SAMPLE_APPLICATION)){const input=form.elements.namedItem(name);if(input)input.value=""}byId("warningStatement").value=STANDARD_WARNING}
function setWorkflow(mode,{announce=true}={}){
  workflowMode=mode==="scan"?"scan":"compare";const scanOnly=workflowMode==="scan";
  compareWorkflowButton.classList.toggle("active",!scanOnly);compareWorkflowButton.setAttribute("aria-pressed",String(!scanOnly));scanWorkflowButton.classList.toggle("active",scanOnly);scanWorkflowButton.setAttribute("aria-pressed",String(scanOnly));
  applicationSection.classList.toggle("hidden",scanOnly);for(const input of applicationSection.querySelectorAll("input,textarea"))input.disabled=scanOnly;fileInput.multiple=!scanOnly;const batchHint=uploadArea.querySelector(".batch-hint");if(batchHint)batchHint.classList.toggle("hidden",scanOnly);
  if(scanOnly&&selectedFiles.length>1)selectFiles([selectedFiles[0]]);workflowHelp.textContent=scanOnly?"Upload one label to transcribe the five visible fields. No application comparison will be performed.":"Enter the approved application values, then compare them with the uploaded label.";actionHelp.textContent=scanOnly?"Scan-only results can contain AI or OCR mistakes. Confirm them against the image.":"Automated check only. A compliance agent makes the final decision.";buttonLabel.textContent=submitLabel();
  resultsSection.classList.add("hidden");if(announce)setNotice(scanOnly?"Scan-only mode reports what the system can read. It does not approve the label or replace compliance review.":"");
}
function resetForNewLabel(){clearSelectedFiles();clearApplicationFields();resultsSection.classList.add("hidden");resultFields.replaceChildren();scanAdvisory.classList.add("hidden");hideError();setNotice("");serverTime.textContent="—";roundTripTime.textContent="—";fileInput.focus();uploadArea.scrollIntoView({behavior:"smooth",block:"center"})}

async function loadWorkingSample(){
  hideError();setNotice("");setWorkflow("compare",{announce:false});const sampleButton=byId("sampleButton");sampleButton.disabled=true;
  try{const response=await fetch("./assets/sample-label.png?v=4");if(!response.ok)throw new Error(`HTTP ${response.status}`);const blob=await response.blob();selectFiles([new File([blob],"sample-label.png",{type:"image/png"})]);for(const [field,value] of Object.entries(SAMPLE_APPLICATION)){const input=form.elements.namedItem(field);if(input)input.value=value}setNotice("Working sample loaded. Select “Verify label” to run the complete check.");submitButton.focus()}
  catch(error){showError(`The sample label could not be loaded: ${error.message}`)}finally{sampleButton.disabled=false}
}
function setServiceState(state,message){serviceStatus.className=`service-status ${state}`;serviceStatus.querySelector(".status-text").textContent=message;serviceReady=state==="ready";submitButton.disabled=!serviceReady}
function updateModeHelp(){modeHelp.textContent=extractionMode.value==="vision"?visionProvider==="gemini"?"Accurate uses Gemini's free-tier vision and may take 5–15 seconds.":"Accurate uses AI vision for difficult real-world photos and may take 5–15 seconds.":localAvailable&&visionAvailable?"Fast uses bounded local OCR and targets a result in under five seconds.":localAvailable?"Fast local OCR is available. Accurate AI vision requires a configured provider.":"No reading mode is currently available."}
async function checkService(){const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),8000);try{const response=await fetch("/ready",{signal:controller.signal,cache:"no-store"}),data=await response.json();localAvailable=Boolean(data?.extraction?.available_modes?.local);visionAvailable=Boolean(data?.extraction?.vision_provider_configured);visionProvider=data?.extraction?.vision_provider??null;const localOption=extractionMode.querySelector('option[value="local"]'),visionOption=extractionMode.querySelector('option[value="vision"]');localOption.disabled=!localAvailable;visionOption.disabled=!visionAvailable;if(!localAvailable&&visionAvailable)extractionMode.value="vision";else if(!visionAvailable)extractionMode.value="local";updateModeHelp();response.ok&&data.status==="ready"?setServiceState("ready","Verification service ready"):setServiceState("unavailable","Verification unavailable")}catch{localAvailable=false;visionAvailable=false;visionProvider=null;updateModeHelp();setServiceState("unavailable","Verification unavailable")}finally{clearTimeout(timeout)}}
function setLoading(active,startedAt=null){submitButton.disabled=active||!serviceReady;loading.classList.toggle("hidden",!active);if(active){loadingElapsed.textContent="Elapsed: 0.0 seconds";timerId=setInterval(()=>loadingElapsed.textContent=`Elapsed: ${((performance.now()-startedAt)/1000).toFixed(1)} seconds`,100)}else if(timerId!==null){clearInterval(timerId);timerId=null}buttonLabel.textContent=active?workflowMode==="scan"?"Scanning…":"Verifying…":submitLabel()}

function createValueBlock(label,value){const wrapper=document.createElement("div"),term=document.createElement("dt"),description=document.createElement("dd");term.textContent=label;description.textContent=value??"Not detected";wrapper.append(term,description);return wrapper}
function createComparisonRows(comparison){const fragment=document.createDocumentFragment();for(const [field,entry] of Object.entries(comparison||{})){const status=STATUS_LABELS[entry?.status]?entry.status:"needs_review",row=document.createElement("article"),name=document.createElement("div"),values=document.createElement("dl"),badge=document.createElement("span");row.className="field-result";name.className="field-name";name.textContent=FIELD_LABELS[field]||field.replaceAll("_"," ");values.className="field-values";values.append(createValueBlock("Expected",entry?.expected),createValueBlock("Detected",entry?.actual));badge.className=`badge ${status}`;badge.textContent=STATUS_LABELS[status];row.append(name,values,badge);fragment.append(row)}return fragment}
function statusCounts(results){const counts={pass:0,fail:0,needs_review:0};for(const result of results)for(const entry of Object.values(result.comparison||{})){const status=STATUS_LABELS[entry?.status]?entry.status:"needs_review";counts[status]+=1}return counts}
function renderResults(results,processingTimeMs,totalRequestMs){
  const completed=results.filter(result=>result.comparison),errors=results.filter(result=>result.error),counts=statusCounts(completed);resultFields.replaceChildren();
  for(const result of results){const group=document.createElement("section");group.className="batch-group";if(results.length>1){const title=document.createElement("h3");title.className="batch-title";title.textContent=result.filename||"Uploaded label";group.append(title)}if(result.error){const error=document.createElement("div");error.className="batch-error";error.textContent=result.error.detail||"This file could not be processed.";group.append(error)}else group.append(createComparisonRows(result.comparison));resultFields.append(group)}
  let overall="pass",heading=results.length>1?"Batch verification complete":"All fields verified",icon="✓";if(counts.fail>0||errors.length){overall="fail";heading=errors.length?"Some labels need attention":"Differences found";icon="×"}else if(counts.needs_review>0){overall="needs_review";heading="Manual review needed";icon="!"}
  resultSummary.className=`result-summary ${overall}`;summaryLabel.textContent="Verification result";resultsHeading.textContent=heading;summaryIcon.textContent=icon;resultCounts.textContent=`${counts.pass} passed · ${counts.fail} failed · ${counts.needs_review} need review${errors.length?` · ${errors.length} file errors`:""}`;scanAdvisory.classList.add("hidden");fieldResultsHeading.textContent="Field comparison";fieldResultsHelp.textContent="Application values are expected; label text is detected.";serverTime.textContent=formatDuration(processingTimeMs);serverTime.classList.toggle("over-target",extractionMode.value==="local"&&processingTimeMs>5000&&results.length===1);roundTripTime.textContent=formatDuration(totalRequestMs);resultsSection.classList.remove("hidden");resultsSection.focus({preventScroll:true});resultsSection.scrollIntoView({behavior:"smooth",block:"start"})
}
function createScanRows(extracted){const fragment=document.createDocumentFragment();for(const field of Object.keys(FIELD_LABELS)){const row=document.createElement("article"),name=document.createElement("div"),values=document.createElement("dl");row.className="field-result scan-result";name.className="field-name";name.textContent=FIELD_LABELS[field];values.className="field-values";values.append(createValueBlock("Detected from label",extracted?.[field]));row.append(name,values);fragment.append(row)}return fragment}
function renderScanResult(data,totalRequestMs){
  const extracted=data?.extracted||{};
  const found=Object.values(extracted).filter(Boolean).length;
  const processingTimeMs=data?.processing_time_ms;
  resultFields.replaceChildren(createScanRows(extracted));
  resultSummary.className="result-summary scan";
  summaryLabel.textContent="Scan result";
  resultsHeading.textContent="Label scan complete";
  summaryIcon.textContent="✓";
  resultCounts.textContent=`${found} of ${Object.keys(FIELD_LABELS).length} fields detected · review the transcription below`;
  scanAdvisoryText.textContent=data?.advisory||"AI and OCR can misread label text. Confirm these values against the uploaded image.";
  scanAdvisory.classList.remove("hidden");
  fieldResultsHeading.textContent="Detected label fields";
  fieldResultsHelp.textContent="These values were transcribed from the image and were not compared with an application.";
  serverTime.textContent=formatDuration(processingTimeMs);
  serverTime.classList.toggle("over-target",extractionMode.value==="local"&&processingTimeMs>5000);
  roundTripTime.textContent=formatDuration(totalRequestMs);
  resultsSection.classList.remove("hidden");
  resultsSection.focus({preventScroll:true});
  resultsSection.scrollIntoView({behavior:"smooth",block:"start"});
}
async function responseError(response){let message=`The server returned HTTP ${response.status}.`;if(response.headers.get("content-type")?.includes("application/json"))try{const payload=await response.json();if(typeof payload.detail==="string")message=payload.detail}catch{}return message}

fileInput.addEventListener("change",()=>selectFiles(fileInput.files));extractionMode.addEventListener("change",updateModeHelp);compareWorkflowButton.addEventListener("click",()=>setWorkflow("compare"));scanWorkflowButton.addEventListener("click",()=>setWorkflow("scan"));byId("removeFileButton").addEventListener("click",()=>{clearSelectedFiles();fileInput.focus()});byId("sampleButton").addEventListener("click",loadWorkingSample);
for(const name of ["dragenter","dragover"])uploadArea.addEventListener(name,event=>{event.preventDefault();uploadArea.classList.add("is-dragging")});for(const name of ["dragleave","drop"])uploadArea.addEventListener(name,event=>{event.preventDefault();uploadArea.classList.remove("is-dragging")});uploadArea.addEventListener("drop",event=>selectFiles(event.dataTransfer.files));
byId("verifyAnotherButton").addEventListener("click",resetForNewLabel);
form.addEventListener("submit",async event=>{
  event.preventDefault();hideError();setNotice("");if(!serviceReady){showError("The verification service is not ready. Refresh the page and try again.");return}if(!form.reportValidity())return;if(!selectedFiles.length){showError("Choose at least one label image first.");fileInput.focus();return}
  const scanOnly=workflowMode==="scan",isBatch=!scanOnly&&selectedFiles.length>1,accurate=extractionMode.value==="vision",uploadFiles=accurate?selectedFiles:await Promise.all(selectedFiles.map(optimizeLocalUpload)),requestBody=new FormData();if(isBatch)uploadFiles.forEach(file=>requestBody.append("files",file));else requestBody.append("file",uploadFiles[0]);if(!scanOnly)requestBody.append("application_data",JSON.stringify(applicationPayload()));requestBody.append("extraction_mode",extractionMode.value);resultsSection.classList.add("hidden");loadingHeading.textContent=scanOnly?"Reading the label":"Reading and comparing the label";loadingMessage.textContent=isBatch?`Processing ${selectedFiles.length} labels in ${accurate?"Accurate":"Fast"} mode. Batch reviews take longer overall.`:accurate?"Accurate AI vision can take up to 40 seconds when a retry is needed.":"Fast local OCR targets a result in under five seconds.";const startedAt=performance.now();setLoading(true,startedAt);const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),isBatch?180000:accurate?50000:30000),endpoint=scanOnly?"/scan":isBatch?"/verify/batch":"/verify";
  try{const response=await fetch(endpoint,{method:"POST",body:requestBody,signal:controller.signal}),totalRequestMs=performance.now()-startedAt;if(!response.ok){showError(await responseError(response),totalRequestMs);return}const data=await response.json();if(scanOnly){renderScanResult(data,totalRequestMs);return}const results=isBatch?data.results:[data],processing=isBatch?data.progress?.processing_time_ms:data.processing_time_ms;if(!Array.isArray(results)||!results.length){showError("The server returned no verification results.",totalRequestMs);return}renderResults(results,processing,totalRequestMs)}catch(error){showError(error.name==="AbortError"?accurate?"Accurate mode timed out after retrying. Try again or choose Fast mode.":"Verification timed out. Try a smaller image or fewer batch files.":`The request could not reach the service: ${error.message}`,performance.now()-startedAt)}finally{clearTimeout(timeout);setLoading(false)}
});
window.addEventListener("beforeunload",clearPreviewUrl);setWorkflow("compare",{announce:false});checkService();
