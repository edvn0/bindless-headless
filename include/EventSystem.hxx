#include <glm/glm.hpp>

#include "Types.hxx"

struct Event {
    virtual ~Event() = default;
    bool handled = false;
};

struct KeyPressedEvent : Event {
    i32 key;
    i32 scancode;
    i32 mods;
};

struct KeyReleasedEvent : Event {
    i32 key;
    i32 scancode;
    i32 mods;
};

struct MouseButtonPressedEvent : Event {
    i32 button;
    i32 mods;
};

struct MouseButtonReleasedEvent : Event {
    i32 button;
    i32 mods;
};

struct CursorMovedEvent : Event {
    glm::vec2 position;
    glm::vec2 delta;
};

struct ScrollEvent : Event {
    f32 x_offset;
    f32 y_offset;
};

struct CharInputEvent : Event {
    u32 codepoint;
};

struct GamepadButtonPressedEvent:  Event {
    i32 button;
};

class EventDispatcher {
public:
    EventDispatcher(Event &event) : ev(event) {}

    template<typename T, typename F>
    void dispatch(F &&func) {
        if (ev.handled)
            return;

        if (auto *e = dynamic_cast<T *>(&ev)) {
            ev.handled = func(*e);
        }
    }

private:
    Event &ev;
};

class EventSystem {
public:
    using EventCallback = std::function<void(Event &)>;

    auto set_event_callback(EventCallback cb) { callback = std::move(cb); }

    auto push_event(std::unique_ptr<Event> event) {
        if (callback) {
            callback(*event);
        }
    }

private:
    EventCallback callback;
};
