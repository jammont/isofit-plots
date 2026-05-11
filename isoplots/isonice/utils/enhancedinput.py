import asyncio
import inspect
import re
from pathlib import Path

from nicegui import ui


class EnhancedInput:
    def __init__(self, label=None, on_change=None, options=[], default="Browse", vertical=False, animated=True, opts_close=True):
        """
        Parameters
        ----------
        label : str, default=None
            Label to set for the input field
        on_change : func, default=None
            Calls this function when the dropdown closes if the input value has changed
        options : list, default=[]
            List of options to set
        default : "Browse", "Options", default="Browse"
            Default tab to start with
        vertical : bool, default=False
            Use the vertical styling for tabs
        animated : bool, default=True
            Makes the dropdown animated. If disabled, the dropdown remains permanently
            open
        opts_close : bool, default=True
            Clicking on an option auto-closes the dropdown and calls the on_change
            If False, the dropdown stays open and only calls the on_change once it
            closes by moving the mouse out of it
        """
        # Track if the dropdown is open
        self._opened = False

        # Track the last value, only call on_change when the new is different than the old
        self._lastValue = None

        # Track the current table to prevent needless refreshes
        self._currTable = None

        self.default = default
        self.animated = animated
        self.on_change = on_change
        self.opts_close = opts_close

        with (ui.column()
            .classes("relative")
            .on("mouseleave", self.toggleDropdown)
        ) as self.wrapper:
            self.input = ui.input(
                label = label,
                value = "",
                on_change = self.glob,
            ).classes("w-full") \
            .on("click", self.open) \
            .on("keydown", self.keydown)

            with ui.element('div').classes('w-full bg-black shadow-lg rounded') as self.dropdown:
                if self.animated:
                    self.dropdown.classes('transition-all duration-500 origin-top scale-y-0 overflow-hidden absolute left-0 top-14 z-50')
                    # self.dropdown.style('transform: scaleY(0); transform-origin: top;')

                if vertical:
                    with ui.splitter(value=10).classes('w-full h-full') as splitter:
                        with splitter.before:
                            self._construct("tabs")
                            self.tabs.props("vertical")
                        with splitter.after:
                            self._construct("panels")
                else:
                    self._construct("tabs")
                    self._construct("panels")

                self.set_options(options)

    def _construct(self, component):
        """
        Easy constructor for the tabs components

        Parameters
        ----------
        component : "tabs" | "panels"
            Which component to construct
        """
        if component == "tabs":

            with ui.tabs().classes('w-full') as self.tabs:
                ui.tab('Browse').classes("flex-1")
                ui.tab('Options').classes("flex-1")
                ui.button(icon="all_out", on_click=self.resolve) \
                    .classes("h-full") \
                    .props("outline dense") \
                    .tooltip("Resolve the current path")

        elif component == "panels":

            with ui.tab_panels(self.tabs, value=self.default).classes('w-full'):
                with ui.tab_panel("Browse").classes("p-0"):
                    self.grid = ui.aggrid({
                        "columnDefs": [
                            # {"field": "path"}
                            {"headerName": "path", "field": "path"}
                        ],
                        "rowData": [{}]
                    }, theme="balham-dark"
                    ).classes(
                        "hide-header"
                    ).on('cellClicked',
                        lambda event: self.appendSearch(event.args["value"])
                    )
                self.opts = ui.tab_panel("Options").classes("p-0")

    async def keydown(self, event):
        """
        Handler for a keydown event, ie. typing in the input field

        Parameters
        ----------
        event : GenericEventArguments
            Key event
        """
        key = event.args["key"]
        shift = event.args["shiftKey"]

        if key == "Enter":
            self.toggleDropdown()
            return
        elif key == "Backspace" and shift:
            pass
            # REVIEW: Doesn't seem to work
            # self.appendSearch("")

        await self.open(tab="Browse")

    @property
    def value(self):
        """
        Passthrough attribute to self.input.value
        """
        return self.input.value

    def classes(self, classes):
        """
        Passthrough to self.wrapper.classes
        """
        self.wrapper.classes(classes)
        return self

    def props(self, props):
        """
        Passthrough to self.wrapper.props
        """
        self.wrapper.props(props)
        return self

    def set_value(self, value):
        """
        Passthrough to self.input.set_value
        """
        self.input.set_value(value)
        self._call_on_change()

    def set_options(self, options):
        """
        Sets the options list

        Parameters
        ----------
        options : list
            List of options to set
        """
        self.options = options

        data = [{"path": opt} for opt in options]

        self.opts.clear()
        with self.opts:
            ui.aggrid({
                "columnDefs": [
                    {"field": "path"}
                ],
                "rowData": data
            }, theme="balham-dark") \
            .classes("hide-header") \
            .on('cellClicked',
                lambda event: self._opt_set_input(event.args["value"])
            )

    def _opt_set_input(self, value):
        """
        Handler for the options list items being clicked

        Parameters
        ----------
        value : str
            Value from the options list that was clicked
        """
        self.input.set_value(value)

        if self.opts_close:
            self.toggleDropdown()

    def set_error(self, message):
        """
        Sets the error message for the input

        Parameters
        ----------
        message : str
            Message to set
        """
        self._error_message = message
        self.input.props(f'error error-message="{message}"')
        # self.input.update()

    def clear_error(self):
        """
        Clears the error message for the input
        """
        self._error_message = None
        self.input.props("")
        #
        # # Get current props
        # current = self.input._props
        #
        # # Remove 'error' and 'error-message="..."'
        # cleaned = re.sub(r'error-message="[^"]*"\s*', '', current)
        # cleaned = re.sub(r'\berror\b\s*', '', cleaned)
        #
        # self.input.props(cleaned.strip())
        # # self.input.update()

    def set_tab(self, tab):
        """
        Sets the active tab

        Parameters
        ----------
        tab : str
            Tab to set
        """
        self.tabs.set_value(tab)

    async def open(self, tab=None):
        """
        Opens the dropdown

        Parameters
        ----------
        tab : str, default=None
            Changes to the tab
        """
        if tab:
            self.set_tab(tab)

        if self.animated:
            self.toggleDropdown(True)

        await self.glob()

    def toggleDropdown(self, open=False):
        """
        Opens/closes the dropdown using its animation

        Parameters
        ----------
        open : bool, default=False
            Open/close the dropdown
        """
        if self.animated:
            if open:
                self._opened = True
                self.dropdown.style('transform: scaleY(1);')
            elif self._opened:
                self._opened = False
                self.dropdown.style('transform: scaleY(0);')
                self._call_on_change()
        else:
            self._call_on_change()

    async def glob(self):
        """
        Globs the filesystem at the current search path, placing directories before
        files

        Parameters
        ----------
        path : ValueChangeEventArguments
            Event when the search string changes
        """
        path = Path(self.input.value)
        if not path.exists():
            glob = path.parent.glob(f"{path.name}*")
        else:
            if path.is_file():
                path = path.parent
            glob = path.glob("*")

        dirs = []
        files = []
        for file in glob:
            if file.is_dir():
                dirs.append(file.name + "/")
            else:
                files.append(file.name)

        # If the parent is the base dir, don't include the /
        parent = str(path.parent)
        if parent == ".":
            parent = ""
        else:
            parent += "/"

        data = ["../"] + sorted(dirs) + sorted(files)
        auto = [f"{parent}{file}" for file in data]

        # self.input.set_autocomplete(auto)

        if self._currTable != data:
            self._currTable = data

            self.grid.options['rowData'] = [{"path": file} for file in data]
            # self.grid.update()

    def appendSearch(self, path):
        """
        Appends a string to the current search string

        Parameters
        ----------
        path : str
            Path to append
        """
        current = Path(self.input.value)
        if not current.exists() or current.is_file():
            current = current.parent

        if path == "../":
            # In order to get parents properly when at base directory, resolve it
            if str(current) == ".":
                current = current.resolve()
            new = current.parent
        else:
            new = current / path

        if new.is_dir():
            new = f"{new}/"

        self.input.set_value(str(new))

    def resolve(self):
        """
        Resolves the current input string
        """
        new = Path(self.input.value).resolve()

        if new.is_dir():
            new = f"{new}/"

        self.input.set_value(str(new))

    def _call_on_change(self):
        """
        Calls the on_change function if the value has changed
        """
        current = self.input.value

        if not Path(current).exists():
            self.set_error("Path does not exist")

        if self.on_change and current != self._lastValue:
            self._lastValue = current

            # asyncio.create_task(self._wrap_on_change(current))
            ui.run_task(self._wrap_on_change(current))

    async def _wrap_on_change(self, value):
        """
        Wraps the on_change function to await if it is an async
        """
        result = self.on_change(value)
        if inspect.isawaitable(result):
            await result


def _root():
    with ui.column().classes("w-[50vw]"):
        EnhancedInput("Options",
            default="Options",
            on_change=lambda s: print(s),
            options=[
                "/test/path",
                "/another/path/to/add",
                "."
            ]
        ).classes("w-full")
        EnhancedInput("Browse").classes("w-full")
        EnhancedInput("Not animated", animated=False).classes("w-full")


# For debugging purposes
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(_root, dark=True)
