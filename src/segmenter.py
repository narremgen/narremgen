"""
narremgen.segmenter
==================
Interactive GUI for manual segmentation of narrative source texts.

Provides a lightweight Tkinter application to load raw text, insert and inspect
segment markers, and export segments as CSV and per-extract text files ready
for downstream use in the Narremgen pipeline.
"""

import os
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox

class SimpleSegmenterApp:
    def __init__(self, root):
        """
        Initialize the graphical interface for manual text segmentation and export.

        This constructor builds a lightweight Tkinter application designed to
        manually segment narrative text into discrete extracts. The interface
        allows users to open a text file, insert or remove segmentation markers,
        visualize markers in color, and export segments to CSV and text files.

        Parameters
        ----------
        root : tkinter.Tk
            The root Tkinter window provided by the caller.

        Attributes
        ----------
        text_area : tkinter.Text
            The main text widget displaying the loaded content with syntax highlighting.
        marker_token : str
            The token string used to mark segment boundaries (default '>||<').
        filename : str | None
            The path of the currently opened text file, or None if no file is loaded.

        Notes
        -----
        - This application is intended for quick visual inspection and segmentation of narratives.
        - Each marker ('>||<') indicates a new segment boundary when exporting.
        - The user interface includes buttons for loading, auto-marking, saving, and exporting.
        """

        self.root = root
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("Segmenter")

        self._themes = {
            "light": {"bg": "#f6f7f9", "text_bg": "#ffffff", "text_fg": "#111827", "cursor": "#111827",
                    "btn_bg": "#dbeafe", "btn_fg": "#0f172a", "btn_abg": "#bfdbfe", "btn_afg": "#0f172a",
                    "m_bg": "#fde047", "m_fg": "#111827", "mc_bg": "#fb923c", "mc_fg": "#111827"},
            "dark":  {"bg": "#1e1e1e", "text_bg": "#000000", "text_fg": "#ffffff", "cursor": "#00ff00",
                    "btn_bg": "#374151", "btn_fg": "#f9fafb", "btn_abg": "#4b5563", "btn_afg": "#f9fafb",
                    "m_bg": "#fde047", "m_fg": "#111827", "mc_bg": "#fb923c", "mc_fg": "#111827"},
        }
        self._theme = "light"
        self._apply_theme(self._theme)
        self._apply_theme(self._theme)

        top = root if isinstance(root, (tk.Tk, tk.Toplevel)) else root.winfo_toplevel()
        top.bind_all("<Control-Shift-d>", lambda e: self._set_theme("dark"))
        top.bind_all("<Control-Shift-l>", lambda e: self._set_theme("light"))
        top.bind_all("<Control-Shift-t>", lambda e: self._toggle_theme())

        self._buttons = []

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill="x", padx=5, pady=(5, 5))

        text_frame = tk.Frame(root)
        text_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        scrollbar = tk.Scrollbar(text_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.text = tk.Text(
            text_frame,
            wrap="word",
            undo=True,
            yscrollcommand=scrollbar.set
        )
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text.yview)

        self.text.tag_config("marker", background=self._themes["light"]["m_bg"], foreground=self._themes["light"]["m_fg"])
        self.text.tag_config("marker_cursor", background=self._themes["light"]["mc_bg"], foreground=self._themes["light"]["mc_fg"])

        self.text.bind("<KeyPress>", self._on_text_keypress)
        self.text.bind("<ButtonRelease-1>", lambda e: self._highlight_markers())
        self.text.bind("<KeyRelease>", lambda e: self._highlight_markers())

        for label, cmd in [
            ("Annuler", self.text.edit_undo),
            ("Charger le texte", self.load_text),
            ("Auto marquer", self.auto_mark),
            ("Insérer Marqueur", self._insert_marker),
            ("Supprimer Marqueur", self._delete_marker),
            ("Ajouter Ligne", self.insert_newline),
            ("Sauvegarder CSV", self.save_csv),
            ("Sauvegarder texte marqué", self.save_marked_text),
            ("Désélectionner", lambda: self.text.tag_remove(tk.SEL, "1.0", tk.END)),
        ]:
            b = tk.Button(
                btn_frame, text=label, command=cmd,
                bg=self._btn_bg, fg=self._btn_fg,
                activebackground=self._btn_abg, activeforeground=self._btn_afg,
                relief="groove", bd=1, padx=8, pady=2, cursor="hand2"
            )
            b.pack(side="left", padx=3, pady=3)
            self._buttons.append(b)

        self._set_theme(self._theme)
        self._highlight_markers()

    def _apply_theme(self, name):
        t = self._themes[name]
        self._bg = t["bg"]; self._text_bg = t["text_bg"]; self._text_fg = t["text_fg"]; self._cursor = t["cursor"]
        self._btn_bg = t["btn_bg"]; self._btn_fg = t["btn_fg"]
        self._btn_abg = t["btn_abg"]; self._btn_afg = t["btn_afg"]
        self.root.configure(bg=self._bg)

    def _set_theme(self, name):
        if name not in self._themes:
            return
        self._theme = name
        t = self._themes[name]
        self.root.configure(bg=t["bg"])
        for w in self.root.winfo_children():
            if isinstance(w, tk.Frame):
                w.configure(bg=t["bg"])
        for b in getattr(self, "_buttons", []):
            b.configure(
                bg=t["btn_bg"], fg=t["btn_fg"],
                activebackground=t["btn_abg"], activeforeground=t["btn_afg"]
            )
        self.text.configure(bg=t["text_bg"], fg=t["text_fg"], insertbackground=t["cursor"])
        self.text.tag_config('marker', background=t["m_bg"], foreground=t["m_fg"])
        self.text.tag_config('marker_cursor', background=t["mc_bg"], foreground=t["mc_fg"])
        self._highlight_markers()

    def _toggle_theme(self):
        self._set_theme("dark" if self._theme == "light" else "light")

    def _on_text_keypress(self, e):
        nav = {
            "Left","Right","Up","Down",
            "Home","End","Prior","Next",
            "Page_Up","Page_Down"
        }
        mods = {"Shift_L","Shift_R","Control_L","Control_R","Alt_L","Alt_R"}

        ctrl = (e.state & 0x4) != 0
        shift = (e.state & 0x1) != 0
        k = (e.keysym or "").lower()

        if ctrl and k == "m" and not shift:
            self._insert_marker()
            return "break"

        if ctrl and k == "m" and shift:
            self._delete_marker()
            return "break"

        if ctrl and k == "j":
            self._goto_next_marker()
            return "break"

        if ctrl and k == "k":
            self._goto_prev_marker()
            return "break"

        if e.keysym in nav or e.keysym in mods:
            return

        if ctrl and k in {"c", "a", "f"}:
            return

        return "break"

    def _goto_next_marker(self):
        cur = self.text.index(tk.INSERT)
        nxt = self.text.search(">||<", cur, forwards=True, stopindex=tk.END)
        if not nxt:
            return
        self.text.mark_set(tk.INSERT, f"{nxt}+4c")
        self.text.see(tk.INSERT)
        self._highlight_markers()

    def _goto_prev_marker(self):
        cur = self.text.index(tk.INSERT)
        prv = self.text.search(">||<", cur, backwards=True, stopindex="1.0")
        if not prv:
            return
        self.text.mark_set(tk.INSERT, f"{prv}+4c")
        self.text.see(tk.INSERT)
        self._highlight_markers()

    def load_text(self):
        """
        Load a plain text file into the editor and refresh segment markers.

        This function opens a file-selection dialog limited to `.txt` files,
        reads the chosen file, and displays its content inside the main text
        widget. Once the text is loaded, any segmentation markers present are
        highlighted automatically.

        Parameters
        ----------
        self : SimpleSegmenterApp
            The current instance of the segmentation interface.

        Returns
        -------
        None
            The function updates the text widget in place and refreshes highlights.

        Notes
        -----
        - The function resets the current text area before loading new content.
        - Only plain UTF-8 text files are supported.
        - The markers '>||<' are highlighted automatically after loading.
        """

        path = filedialog.askopenfilename(filetypes=[("Text Files","*.txt"),("All Files","*.*")])
        if not path: return
        content = open(path, encoding='utf-8').read()
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, content)
        self._highlight_markers()

    def auto_mark(self):
        """
        Automatically insert segmentation markers into a diary-style text.

        This feature analyzes the loaded text heuristically and inserts the marker
        token ('>||<') at probable narrative boundaries. The heuristic currently:
        1. Detects lines that begin with a date or a day name (e.g., "12 June", "Monday").
        2. Splits long paragraphs every ~5 sentences to improve readability.
        3. Rebuilds the text with inserted markers, then refreshes highlighting.

        Parameters
        ----------
        self : SimpleSegmenterApp
            The current instance of the segmentation interface.

        Returns
        -------
        None
            Updates the editor content in place with markers inserted.

        Notes
        -----
        - The algorithm uses simple regex and sentence heuristics rather than NLP parsing.
        - Existing markers are preserved; duplicates are avoided.
        - After auto-marking, the user can adjust markers manually before exporting.
        """
        content = self.text.get('1.0', tk.END)
        date_pattern = re.compile(r'^(\d{1,2} (?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b)', re.MULTILINE)
        content = date_pattern.sub(r'>||< \1', content)
        parts = content.split('>||<')
        rec = ''
        for part in parts:
            seg = part.strip()
            if not seg: continue
            lines = seg.split('\n',1)
            head, body = ('', seg)
            if re.match(r'^\d{1,2} (?:mai|juin|juillet|août)\b', lines[0]):
                head = lines[0] + '\n'
                body = lines[1] if len(lines)>1 else ''
            sentences = re.split(r'(?<=[\.\!?])\s+', body)
            new_body = ''
            for i, s in enumerate(sentences,1):
                if not s.strip(): continue
                new_body += s.strip() + ' '
                if i % 5 == 0:
                    new_body += '>||< '
            rec += '>||< ' + head + new_body.strip() + '\n'
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, rec)
        self._highlight_markers()

    def _insert_marker(self):
        """
        Insert a segmentation marker token at the current cursor position.

        This function inserts the marker string (default '>||<') into the text
        widget at the current insertion point. It is typically triggered by a
        button click or keyboard shortcut to manually define segment boundaries.

        Parameters
        ----------
        self : SimpleSegmenterApp
            The current instance of the segmentation interface.

        Returns
        -------
        None
            Inserts the marker into the text widget and refreshes the highlight.

        Notes
        -----
        - The marker visually separates text segments for later export.
        - After insertion, the marker is immediately highlighted using `_highlight_markers()`.
        - Manual markers can complement or refine those generated automatically
        by the `auto_mark()` function.
        """

        pos = self.text.index(tk.INSERT)
        try:
            sel = self.text.get(pos, f"{pos}+4c")
        except tk.TclError:
            sel = ''
        if sel == '>||<':
            return
        self.text.insert(pos, '>||< ')
        self._highlight_markers()

    def _delete_marker(self):
        """
        Delete the nearest segmentation marker surrounding the cursor position.

        This function removes the closest marker token ('>||<') to the text cursor.
        It is designed to let users quickly correct or adjust automatic or manual
        segment boundaries. After deletion, the highlighting is refreshed to reflect
        the current marker layout.

        Parameters
        ----------
        self : SimpleSegmenterApp
            The current instance of the segmentation interface.

        Returns
        -------
        None
            Updates the text widget content after removing one marker occurrence.

        Notes
        -----
        - The search is bidirectional: the function looks before and after the cursor
        to find the nearest marker.
        - If no marker is found, the function performs no action.
        - Highlights are automatically refreshed after modification.
        """

        marker = ">||<"
        pos = self.text.index(tk.INSERT)

        before = self.text.search(marker, pos, backwards=True, stopindex='1.0')
        if before:
            end_before = f"{before}+{len(marker)}c"
            if self.text.compare(pos, ">=", before) and self.text.compare(pos, "<=", end_before):
                self.text.delete(before, end_before)
                self._highlight_markers()
                return

        after = self.text.search(marker, pos, forwards=True, stopindex=tk.END)
        if after:
            end_after = f"{after}+{len(marker)}c"
            if self.text.compare(pos, ">=", after) and self.text.compare(pos, "<=", end_after):
                self.text.delete(after, end_after)
                self._highlight_markers()
                return

        big = 10**9
        dist_before = big
        dist_after = big

        if before:
            dist_before = abs(int(self.text.count(before, pos)[0]))
        if after:
            dist_after = abs(int(self.text.count(pos, after)[0]))

        if dist_before == big and dist_after == big:
            messagebox.showinfo("Pas de marqueur", "Aucun marqueur trouvé à proximité.")
            return

        if dist_before <= dist_after and before:
            target = before
        else:
            target = after

        end = f"{target}+{len(marker)}c"
        self.text.delete(target, end)
        self._highlight_markers()
    
    def insert_newline(self):
        """
        Insert a blank line in the text area at the current cursor position.

        This simple editing utility adds a new empty line within the text widget,
        allowing the user to visually separate narrative blocks or prepare areas
        for manual annotation.

        Parameters
        ----------
        self : SimpleSegmenterApp
            The current instance of the segmentation interface.

        Returns
        -------
        None
            Inserts a newline into the text widget and refreshes the highlighting.

        Notes
        -----
        - Unlike standard typing, this action can be bound to a GUI button for
        convenience during annotation.
        - After insertion, `_highlight_markers()` is called to maintain consistent
        highlighting of markers in the surrounding text.
        """

        pos = self.text.index(tk.INSERT)
        self.text.insert(pos, '\n\n')

    def save_csv(self):
        """
        Export the segmented text to a CSV file and individual text extracts.

        This function splits the current editor content using the marker token
        (`>||<`) to identify segment boundaries. Each segment is written to a
        CSV file with structural placeholders and also saved as a separate text
        file under a `segments/` subdirectory. The CSV is structured for later
        annotation or integration into the Narremgen pipeline.

        Parameters
        ----------
        self : SimpleSegmenterApp
            The current instance of the segmentation interface.

        Returns
        -------
        None
            The function performs file writing operations and does not return data.

        Output Format
        -------------
        The exported CSV includes the following columns:
        `X`, `name_SN`, `struct_SN`, `name_DE`, `struct_DE`,
        `debut_extrait`, `fin_extrait`.

        Notes
        -----
        - Each segment is saved in a file named `segments/extrait_<basename>_<i>.txt`.
        - If files already exist in the target folder, an error dialog prevents overwriting.
        - This export step facilitates manual labeling and structural enrichment
        of narrative fragments.
        """

        raw = self.text.get('1.0', tk.END)
        segments = [seg.strip() for seg in raw.split('>||<') if seg.strip()]
        if not segments:
            messagebox.showwarning("Aucun segment", "Aucun marqueur trouvé.")
            return
        base_dir = os.path.dirname(getattr(self, 'source_path', '')) or os.getcwd()
        base_name = getattr(self, 'source_basename', 'file')
        csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not csv_path: return
        out_dir = os.path.dirname(csv_path) or os.getcwd()
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        dir_sgmts = os.path.join(out_dir, f"{csv_stem}_segments")
        os.makedirs(dir_sgmts, exist_ok=True)
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X","segment","debut_extrait","fin_extrait","name_SN","struct_SN","name_DE","struct_DE"])
            for i, seg in enumerate(segments, start=1):
                seg_csv = seg.replace("\r\n", "\n")
                debut = seg_csv[:15]
                fin = seg_csv[-15:]
                writer.writerow([i, seg_csv, debut, fin, "", "", "", ""])
                txt_name = f"extrait_{base_name}_{i}.txt"
                txt_path = os.path.join(dir_sgmts, txt_name)
                if os.path.exists(txt_path):
                    messagebox.showerror("Erreur", f"Le fichier '{txt_name}' existe déjà. Annulation.")
                    return
                with open(txt_path, 'w', encoding='utf-8') as tf:
                    tf.write(seg)
        messagebox.showinfo("Sauvegarde CSV",f"{len(segments)} segments enregistrés.\nCSV: {csv_path}\nExtraits: {dir_sgmts}")

    def save_marked_text(self):
        """
        Save the current text content, including all markers, to a new text file.

        This function opens a save dialog that allows the user to write the full
        content of the text editor (including any `>||<` markers) to a plain UTF-8
        text file. It ensures that existing files are not overwritten without
        explicit user confirmation.

        Parameters
        ----------
        self : SimpleSegmenterApp
            The current instance of the segmentation interface.

        Returns
        -------
        None
            The function writes a `.txt` file containing the full editor content.

        Notes
        -----
        - This save option preserves the segmentation markers for later reloading.
        - A confirmation dialog is shown if the target filename already exists.
        - The operation does not alter the text currently displayed in the editor.
        """

        path = filedialog.asksaveasfilename(defaultextension=".txt",filetypes=[("Text","*.txt")])
        if not path: return
        if os.path.exists(path):
            messagebox.showerror("Erreur","Fichier existe déjà !")
            return
        with open(path,'w',encoding='utf-8') as f:
            f.write(self.text.get('1.0',tk.END))
        messagebox.showinfo("Texte enregistré",path)

    def _highlight_markers(self):
        """
        Highlight all marker tokens inside the text widget for better visibility.

        This internal helper scans the text content for occurrences of the marker
        token (default '>||<') and applies a visual highlight using a custom text
        tag. Existing highlights are cleared before reapplying the new ones.

        Parameters
        ----------
        self : SimpleSegmenterApp
            The current instance of the segmentation interface.

        Returns
        -------
        None
            The function updates tag styling and applies highlighting directly
            within the text widget.

        Notes
        -----
        - The highlight uses a configurable color tag (typically yellow background)
        for easy identification of narrative segment boundaries.
        - This function is called automatically after any edit, load, or save
        operation that modifies the text content.
        """

        self.text.tag_remove('marker','1.0',tk.END)
        self.text.tag_remove('marker_cursor','1.0',tk.END)
        idx='1.0'
        while True:
            idx=self.text.search('>||<',idx,stopindex=tk.END)
            if not idx:break
            self.text.tag_add('marker',idx,f"{idx}+4c")
            idx=f"{idx}+1c"
        cur=self.text.index(tk.INSERT)
        if 'marker' in self.text.tag_names(cur):
            start=self.text.search('>||<',cur,backwards=True,stopindex='1.0')
            if start:
                self.text.tag_add('marker_cursor',start,f"{start}+4c")
