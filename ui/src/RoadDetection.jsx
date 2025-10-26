// // ...existing code...
// import React, { useState, useRef} from "react";
// import './UploadFolder.css';
// import { useLocation } from 'react-router-dom';

// const RoadDetection = () => {
//   const [files, setFiles] = useState([]);
//   const [uploading, setUploading] = useState(false);
//   const [progress, setProgress] = useState(0);
//   const [status, setStatus] = useState(null);
//   const [processing, setProcessing] = useState(false);
//   const [outputs, setOutputs] = useState([]); // <- new
//   const [showIframe, setShowIframe] = useState(false);
//   const inputRef = useRef(null);
//   const location = useLocation();

//     const fetchOutputs = async () => {
//     // try {
//     //   const resp = await fetch("http://127.0.0.1:8000/outputs");
//     //   if (!resp.ok) throw new Error(`Server ${resp.status}`);
//     //   const data = await resp.json();
//     //   setOutputs(data.outputs || []);
//     // } catch (err) {
//     //   console.error("load outputs:", err);
//     // }
//     // window.open("http://127.0.0.1:8000/outputs", "_blank", "noopener,noreferrer");
//       const pagePath = encodeURIComponent(location.pathname || '');
//       window.location.href = `http://localhost:8000/outputs`;
//   };


// // const fetchOutputs = async () => {
// //   try {
// //     const resp = await fetch("http://127.0.0.1:8000/outputs");
// //     if (!resp.ok) throw new Error(`Server ${resp.status}`);
// //     const data = await resp.json();
// //     const backendBase = "http://127.0.0.1:8000";
// //     const normalized = (data.outputs || []).map(o => {
// //       let url = o.url || "";
// //       if (url.startsWith("/")) url = `${backendBase}${url}`;            // "/static/outputs/..." -> "http://127.0.0.1:8000/static/outputs/..."
// //       else if (!url.startsWith("http")) url = `${backendBase}/${url}`; // "static/outputs/..." -> "http://127.0.0.1:8000/static/outputs/..."
// //       return { ...o, url };
// //     });
// //     setOutputs(normalized);
// //   } catch (err) {
// //     console.error("load outputs:", err);
// //   }
// // };
// // // ...existing code...

//   const handleFiles = (fileList) => {
//     const arr = Array.from(fileList).filter(f => f.type.startsWith("image/"));
//     setFiles(arr);
//     setStatus(null);
//     setProgress(0);
//   };

//   const onSelectFolder = (e) => {
//     handleFiles(e.target.files);
//   };

//   const onDrop = (e) => {
//     e.preventDefault();
//     if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
//   };

//   const onDragOver = (e) => {
//     e.preventDefault();
//   };

//   const uploadFiles = () => {
//     if (!files.length) return;
//     setUploading(true);
//     setStatus(null);
//     setProgress(0);

//     const form = new FormData();
//     files.forEach((file) => {
//       // preserve folder structure when available (webkitRelativePath)
//       const key = file.webkitRelativePath || file.name;
//       form.append("images", file, key);
//     });

//     // Use XMLHttpRequest to track upload progress
//     const xhr = new XMLHttpRequest();
//     const pagePath = encodeURIComponent(location.pathname || '');
//     xhr.open("POST",`http://localhost:8000/upload`, true);

//     xhr.upload.onprogress = (e) => {
//       if (e.lengthComputable) {
//         const pct = Math.round((e.loaded / e.total) * 100);
//         setProgress(pct);
//       }
//     };

//     xhr.onload = () => {
//       setUploading(false);
//       if (xhr.status >= 200 && xhr.status < 300) {
//         setStatus({ ok: true, message: "Upload complete. Processing images..." });
//         setFiles([]);
//         if (inputRef.current) inputRef.current.value = null;
//       } else {
//         setStatus({ ok: false, message: `Upload failed (${xhr.status})` });
//       }
//     };

//     xhr.onerror = () => {
//       setUploading(false);
//       setStatus({ ok: false, message: "Upload error" });
//     };

//     xhr.send(form);
//     // refresh outputs list after successful processing
//     try { process_image() } catch (e) { /* ignore fetchOutputs errors */ }
//   };

//         const process_image = async () => {
//             setProcessing(true);
//             setStatus(null);

//             try {
//                 const resp = await fetch("http://localhost:8000/process_road", { method: "POST" });
//                 if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
//                 const payload = await resp.json();

//                 setStatus({ ok: true, message: "Processing complete — results ready to view." });
//             } catch (err) {
//                 setStatus({ ok: false, message: `Processing failed: ${err.message}` });
//             } finally {
//                 setProcessing(false);
//             }
//         };

//   return (
//     <div style={{ fontFamily: "Inter, system-ui, sans-serif", maxWidth: 900, margin: "32px auto", padding: 20 }}>
//       <header style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
//         <div aria-hidden="true" style={{ width: 44, height: 44, display: "flex", alignItems: "center", justifyContent: "center", background: "#eef2ff", borderRadius: 8 }}>
//           <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
//             <circle cx="12" cy="12" r="9" stroke="#2dd4bf" strokeWidth="1.2" fill="none" />
//             <path d="M5 15c2-3 6-5 8-5s6 2 8 5" stroke="#6366f1" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
//           </svg>
//         </div>
//         <div>
//           <div style={{ fontWeight: 700 }}>Spectra Space</div>
//           <div style={{ fontSize: 12, color: "#475569" }}>Road Detection — Colorized SAR</div>
//         </div>
//       </header>

//         <div style={{ marginTop: 20 }}>
//           <button onClick={fetchOutputs}>View Results</button>
//         </div>
//     </div>
//   );
// };

// export default RoadDetection;

import React, { useEffect } from 'react';

const RoadDetection = () => {
  useEffect(() => {
    // Redirect to backend endpoint in a new tab or same tab
    window.location.replace('http://localhost:8001/road_output');
  }, []);

  return null; // nothing to render, we immediately redirect
};

export default RoadDetection;
